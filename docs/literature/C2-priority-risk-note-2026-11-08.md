# C2 — дополнительный риск приоритета, abstract-only scout

8 ноября 2026, `/root`, read-only web search. Это заметка к будущему review,
не расширение grid/endpoint и не утверждение нового теоретического результата.

Первичные страницы, прочитанные на уровне abstract:

- Fisher et al., [Statistical and Computational Guarantees for Influence
  Diagnostics, v2](https://arxiv.org/abs/2212.04014v2), 2023: finite-sample и
  computational bounds для influence diagnostics; заявленные приложения GLM
  и attention models. Полные assumptions/theorems здесь не проверены.
- Schioppa et al., [Theoretical and Practical Perspectives on what Influence
  Functions Do](https://deepmind.google/research/publications/56635/), 2023:
  различия training trajectories/parameter divergence ограничивают прогноз
  retraining effects, в том числе при deterministic training. Это первичная
  страница авторской организации, не наш эксперимент.
- Sharchilev et al., [ICML2018](https://proceedings.mlr.press/v80/sharchilev18a.html)
  и Brophy et al., [JMLR2023](https://www.jmlr.org/papers/v24/22-0449.html)
  остаются ближайшими проверенными antecedents по деревьям. Их fixed-structure
  approximation и вычислительные компромиссы уже известны.

Следствие для scope R14: нельзя заявлять первое обнаружение ошибок
линеаризации, нестабильности структуры, общую гарантию influence reliability
или новизну exact conditional replay. Возможный эмпирический вклад ограничен
заранее проверяемым forecast заданного finite-label-response discrepancy,
превосходящим как static, так и простые early-probe baselines. Даже выполнение
discovery gates не устанавливает приоритет, перенос на реальные задачи,
нейросети или double descent. Полный текст наиболее близких работ и отдельная
confirmation нужны до сильного научного claim.

Новые локальные full-text snapshots этим web scout не создавались; byte/commit
hash для этих страниц не заявлен. Existing C2 intake и его acquisition log
сохранены без изменений. Этот документ фиксируется как текстовый input до run.
