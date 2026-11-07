# Состояние на 7 ноября 2026 — R0–R7 операционно закрыты

`scientific_validity=not_assessed`. Критической эскалации по решению 0003 нет.
Локальные claims не являются внешним подтверждением и не заменяют human review.

**Следующая команда только читает состояние. Нового stage нет.**

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

Charged runner wall после сверки ledger: 119.187 с из 86400 с.
`reconciliation_problems` пуст. GPU, платные API и скачивания датасетов: 0.

| Stage | Attempt | Review SHA-256 | Archive SHA-256 |
| --- | --- | --- | --- |
| R2 | `20261107T095343-6bc0c457c7` | `a4e389ae2bf47a1920145439eae84fd9621ff5d6fe407dfbb423762728d79b7b` | `2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055` |
| R3 | `20261107T102511-b1cf6c7be9` | `6c17b3010fdb68ae525962cfec673fac90579a878ca23d99eaaa1edd22756743` | `49700f5b51e730343e57b041c6844903c6ae9f073cb03dd4895d159cf956ff2b` |
| R4 | `20261107T104140-5bc2019cef` | `16ac881579ff2de3f6d821ceaf6b99e502793e34e3646dbfe9b7b83d274e32fe` | `eb360f4c0aa7be3a296dc09ab7dba42f90bcb1db78866dcab4348eef63b51e4b` |
| R5 | `20261107T105955-6e7f800ff1` | `0735cf70372781ed77e5648e3280a3ba5d15d019516c86fd373fd8b8f0e0b338` | `aaed282379380969de2e572303ea9048d5c2002f5f2c5a73fa4d422991c02ea9` |
| R6 | `20261107T111120-40cf0c74f2` | `859cf65cfae08a8e1049b20728665dc2b006c539adce1be65a4fe1bca7f212c8` | `cb059f512515c1b0699db277434c367019198ca7aecba73136b527a418748e74` |
| R7 | `20261107T112024-93842d83aa` | `1bc7dff637fedd7e9326598fe3dfe6b4c82762928b81b4ce5dea5d92f30db51c` | `a8b7d2d41010dc99cb622d12c2eaa6eb0907654f520052ca092e5ed7f22143c0` |

R0 attempt `20261107T070335-5521b09a17`. R1 attempt `20261107T085630-b4345b8c44`.
Записка: `research/stages/R7/article.md`. Внешняя подача не выполнялась.
