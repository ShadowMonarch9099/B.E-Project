"""python -m app.llm models: the chat models each configured key can use (no answer is generated,
nothing is billed). python -m app.llm status: the provider the app will use."""

import asyncio
import json
import sys

from .. import config  # noqa: F401  (loads backend/.env)
from . import get_provider
from .openai_compat import PRESETS, OpenAICompatProvider


async def models(filter_text: str | None) -> None:
    for preset in PRESETS.values():
        p = OpenAICompatProvider(preset)
        if not p.configured():
            print(f"{preset.label}: no key ({preset.key_env} is not set)")
            continue
        try:
            page = await p.client().models.list()
            ids = sorted(m.id for m in page.data)
        except Exception as exc:  # show the failure, never the key
            print(f"{preset.label}: could not list models ({type(exc).__name__})")
            continue
        shown = [i for i in ids if not filter_text or filter_text.lower() in i.lower()]
        mark = "available" if p.model in ids else "NOT in the list"
        print(f"{preset.label}: {len(ids)} models; configured model {p.model} is {mark}")
        for i in shown:
            print(f"  {i}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "models":
        asyncio.run(models(sys.argv[2] if len(sys.argv) > 2 else None))
    elif cmd == "status":
        print(json.dumps(get_provider().status(), indent=2))
    else:
        sys.exit("usage: python -m app.llm [status | models [filter]]")


if __name__ == "__main__":
    main()
