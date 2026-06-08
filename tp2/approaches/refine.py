"""Refinamento iterativo por VLM-crítico (Gemini) — o coração do approach (D3, parte #4).

Fluxo (render-guided, Decisão A): o Gemini semeia (one-shot a partir do alvo) → render+avalia → o
Gemini vê (alvo, render atual) e propõe prompts melhorados → render+avalia → repete R rondas. A
métrica composta (sobre o render real) é o juiz: o pool guarda tudo, e o orquestrador escolhe o
top-3 no fim.

Precisa de uma API key do Gemini em GEMINI_API_KEY (ou GOOGLE_API_KEY). Modelo Flash = grátis.
"""
import io
import os
import time

from ..ranking import composite_score

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # flash-lite mais recente; 500 RPD no tier grátis

# Exemplos few-shot do ESTILO de prompt SD. Assuntos propositadamente sem relação com os 6 alvos,
# para ilustrar o formato sem enviesar o conteúdo gerado.
_STYLE_EXAMPLES = (
    "The examples below show ONLY the desired prompt style/format; their subjects are unrelated and "
    "must NOT influence your content:\n"
    "- extreme close-up cinematic portrait of an elderly tribal chieftain with a dire wolf, dramatic "
    "chiaroscuro, pure black background, towering feathered headdress in teal, orange and scarlet, "
    "weathered skin, piercing amber eyes, high-contrast directional light, 35mm, hyper-detailed, 8k\n"
    "- digital illustration, fantasy ranger in a hood of green leaves with antler branches, glowing "
    "green eyes, drawing a glowing ethereal bow, dark misty forest, green and black palette, mystical "
    "atmosphere, concept art\n"
    "- ultra realistic 3D render, luxury wedding invitation envelope on ivory silk, embossed floral "
    "patterns, champagne gold wax seal, blush roses and white peonies, warm candlelight bokeh, "
    "cinematic depth of field, soft golden light, highly detailed\n"
)

_INSTRUCTION = (
    "You are an expert prompt engineer for Stable Diffusion / LCM text-to-image models, doing "
    "iterative prompt inversion. The TARGET image is the goal. The CURRENT image was rendered from "
    "this prompt:\n\n\"{prompt}\"\n\n"
    "Compare the CURRENT render against the TARGET and find the SPECIFIC things the model got WRONG: "
    "subject or count, shapes, colors, materials, composition and framing, background, lighting, and "
    "especially the medium / art style.\n\n"
    "Then write {n} improved prompts that CORRECT those specific errors — push the model toward the "
    "TARGET by adding, emphasizing or rephrasing the wrong aspects, while KEEPING what already matches. "
    "Write them in Stable Diffusion prompt style: short comma-separated tags/phrases (not sentences), "
    "always stating the medium/style (e.g. photograph, 3D render, oil painting, digital illustration, "
    "anime, sculpture) plus lighting and a few fitting quality boosters.\n\n"
    + _STYLE_EXAMPLES +
    "\nKeep each prompt under 55 words (~75 CLIP tokens): the model ignores anything beyond that, so "
    "be concise and put the most important elements first. Make the {n} prompts diverse. Output ONLY "
    "the prompts, one per line, no numbering, no commentary."
)

_DESCRIBE_INSTRUCTION = (
    "You are an expert prompt engineer for Stable Diffusion / LCM text-to-image models. Look at the "
    "TARGET image and write {n} DIVERSE prompts that, when rendered, would reproduce it as closely as "
    "possible.\n\n"
    "Write them in STABLE DIFFUSION PROMPT STYLE: short comma-separated tags/phrases, not full "
    "sentences. Each prompt should state: the medium/format (photograph, 3D render, oil painting, "
    "digital illustration, anime, sculpture, ...); the main subject and exact count; key colors, "
    "materials, composition and background; lighting and camera (e.g. cinematic lighting, close-up, "
    "wide shot); and a few quality/style boosters typical of SD prompts when fitting (e.g. highly "
    "detailed, sharp focus).\n\n"
    + _STYLE_EXAMPLES +
    "\nKeep each prompt under 55 words (~75 CLIP tokens): the model ignores anything beyond that, so "
    "be concise and put the most important elements first. Make the {n} prompts genuinely different "
    "from each other (especially the medium/style for ambiguous parts). Output ONLY the prompts, one "
    "per line, no numbering, no commentary."
)


class GeminiCritic:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("Falta a API key do Gemini: define GEMINI_API_KEY no ambiente.")
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _to_part(image):
        from google.genai import types
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

    @staticmethod
    def _parse(text, n):
        prompts = []
        for line in (text or "").splitlines():
            line = line.strip().lstrip("-*•0123456789. ").strip().strip('"')
            if len(line) > 3:
                prompts.append(line)
        return prompts[:n]

    def propose(self, target_image, current_image, current_prompt, n: int = 2, retries: int = 3) -> list:
        """Devolve até `n` prompts melhorados, comparando alvo e render atual."""
        self._ensure()
        contents = [
            _INSTRUCTION.format(prompt=current_prompt, n=n),
            "TARGET image:", self._to_part(target_image),
            "CURRENT render:", self._to_part(current_image),
        ]
        for attempt in range(retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=contents)
                return self._parse(resp.text, n)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)  # backoff p/ rate limits

    def describe(self, target_image, n: int = 5, retries: int = 3) -> list:
        """Seed one-shot: escreve `n` prompts só a partir do alvo (sem render)."""
        self._ensure()
        contents = [_DESCRIBE_INSTRUCTION.format(n=n), "TARGET image:", self._to_part(target_image)]
        for attempt in range(retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=contents)
                return self._parse(resp.text, n)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 2)


def refine_approach(critic: GeminiCritic = None, *,
                    seed_fn=None, n_seed: int = 5, beam: int = 3, n_proposals: int = 2,
                    rounds: int = 2, verbose: bool = False):
    """Approach completo: seed + refinamento render-guided por Gemini.

    Seed por omissão = Gemini one-shot (`critic.describe`); passa `seed_fn(ctx) -> list[str]`
    para outro seeder. Devolve todo o pool; o orquestrador ordena o top-3.
    """
    critic = critic or GeminiCritic()
    if seed_fn is None:
        def seed_fn(ctx):
            return critic.describe(ctx.target_image, n=n_seed)

    def approach(ctx):
        scored = {}  # prompt -> (score, image)

        def add(prompt):
            if not prompt or prompt in scored:
                return
            image = ctx.render(prompt)
            metrics = ctx.evaluator.evaluate(ctx.target_image, image)
            scored[prompt] = (composite_score(metrics), image)
            if verbose:
                print(f"   score={scored[prompt][0]:.3f}  {prompt[:70]}")

        if verbose:
            print(f"[{ctx.target_path.name}] seed")
        for p in seed_fn(ctx):
            add(p)

        for r in range(rounds):
            beam_prompts = sorted(scored, key=lambda p: scored[p][0], reverse=True)[:beam]
            if verbose:
                print(f"[{ctx.target_path.name}] ronda {r + 1}/{rounds} (beam top-{beam})")
            for bp in beam_prompts:
                _, bp_img = scored[bp]
                try:
                    proposals = critic.propose(ctx.target_image, bp_img, bp, n=n_proposals)
                except Exception as exc:
                    if verbose:
                        print("   (Gemini falhou:", exc, ")")
                    proposals = []
                for prompt in proposals:
                    add(prompt)

        return list(scored.keys())

    return approach
