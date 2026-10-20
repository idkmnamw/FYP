# HargaLens: a small user evaluation

Use this guide to evaluate whether people can use and correctly interpret the completed dashboard. It does not evaluate model accuracy, actual behaviour change or affordability outcomes. Keep the existing model, cutoff and November registration unchanged.

Start with **5–8 suitable volunteers**, about **20–25 minutes each**, as a practical formative study. Recruit people relevant to your stated audience where possible, such as people who monitor household food spending or review price information. Record their role. If participants are classmates, report this as a convenience sample and do not generalise their responses to professional analysts, policymakers or all Malaysian consumers.

## Before the first session

1. Run the existing dashboard normally. Keep one app version and saved model/snapshot for the first round. Record the Git commit, forecast model version, checkpoint SHA-256 and snapshot week from the app. Do not rerun preparation or adjust the model during sessions.
2. Copy the empty response template to the ignored data folder; keep one row per participant:

   ```bash
   mkdir -p data/evaluation
   cp evaluation/results_template.csv data/evaluation/sessions.csv
   ```

   Fill it in with a spreadsheet editor. Do not repeat the copy command over completed responses. Store completed responses under `data/evaluation/`, not in Git. Use IDs such as P01, not names or contact details. There are no fabricated responses in the template.

3. Pilot the tasks once yourself. Choose and note consistent example items/locations in `task_case_notes`: a supported local series, two Spotlight cases, an available explanation, an unavailable explanation and a historical series without a latest forecast. Use real saved examples. If a required example genuinely does not exist in this snapshot, mark that task not run and explain why; never fabricate a forecast or missing-data example.
4. Use the same six prompts and scoring rules for every participant. Record device/browser and connection conditions. A participant may use the sidebar, tooltips and built-in guide without it counting as moderator assistance.
5. Explain that participation is voluntary, they can stop or skip a question, the prototype uses historical data, and feedback concerns the interface. Ask them to think aloud. Do not teach the correct answers before the tasks. Read only the participant prompts below; keep the answer key to yourself.

## Six participant tasks

Read the bracketed examples using your preselected cases. Start the timer after reading each prompt and stop when the participant completes it or stops. Allow up to **180 seconds per task**. Record navigation time and errors, not typing speed in the feedback sheet.

| Task | Prompt to read aloud |
|---|---|
| T1 — Find a price | “Find [item] in [state/district/shop type]. Tell me its displayed weekly median and unit.” |
| T2 — Interpret the dates | “Which week do these observed prices describe, and which week does the review alert concern? How current is this information?” |
| T3 — Interpret an alert | “Explain this item's review score and next-week status in your own words. What does the 10% surge definition refer to?” |
| T4 — Distinguish signals | “Look at the unusual-pattern flag and the next-week alert. What does each tell you? Could one be present without the other?” |
| T5 — Follow a price story | “Choose [first Spotlight case], then switch to [second case]. Find one influence on the second case's score, explain what it tells you, and download its evidence brief. Check that the item and forecast dates in the brief match your selection.” |
| T6 — Recognise missing support | “Look at [case without a cached explanation] and [historical series without a latest forecast]. What conclusions can you draw from what is missing?” |

For T6, the moderator may open the preselected cases so the task tests interpretation rather than a search for rare examples. State this consistently for everyone; count additional explanatory hints as assistance. If part of T6 is unavailable, mark T6 not run and retain observations about the available part in the notes.

## Moderator answer key and scoring

Do not show this section to participants before completing the tasks.

| Task | Successful interpretation or action |
|---|---|
| T1 | Correct item, location and shop type; correct displayed median and pack/unit. Does not call the median the cheapest shop price. |
| T2 | Correct observed and forecast weeks from the banner. For the current snapshot: 22–28 June 2026 and 29 June–5 July 2026. Recognises a historical demonstration, not today's prices or live warnings. Read the actual banner if a different snapshot is used. |
| T3 | A higher score means greater review priority, not a verified event probability. An alert can be false. The target is an observed next-week local median rise of at least 10%; it is not the score cutoff or a predicted exact price increase. |
| T4 | The forecast concerns a possible next-week surge. The separate detector flags unusual observed patterns. They need not agree; neither flag proves an error or future event. |
| T5 | Switches to the requested case, names an actual cached contribution/direction, and matches the downloaded brief's item/location and forecast dates. Understands that the explanation describes model associations, not proven economic causes. |
| T6 | Missing explanation means no cached explanation, not no influential features or low risk. Missing forecast means insufficient supported information here, not guaranteed stable prices. Does not substitute an older prediction as current. |

Use **U** = correct without moderator help; **H** = correct with moderator help before the time limit; **F** = incorrect, abandoned or timed out; **N** = not run, with a reason. Record result and seconds for each task. Ordinary think-aloud prompts such as “What are you looking for?” are neutral; hints about the answer or navigation count as help. Do not turn a timeout into success because the moderator demonstrates the answer afterwards.

In `critical_misunderstandings`, record any retained belief that scores are event probabilities, the snapshot is live, below-cutoff/missing forecasts guarantee stable prices, or SHAP proves economic causes. Record initial confusions that the participant later resolves under `help_and_issues`, so the two are not conflated. Do not infer comprehension from a positive satisfaction rating.

## Five-minute feedback interview

After the tasks, ask for a rating from **1 = strongly disagree** to **5 = strongly agree**, with **3 = unsure/neutral**:

- Ease of use: “I could find the information I needed.”
- Clarity: “The prices, dates and different signals were understandable.”
- Review usefulness: “The price history, coverage and explanations would help me investigate a flagged case.”

These are three custom formative questions, **not SUS or another validated scale**. Leave a rating blank if the participant declines or it does not apply; never replace it with a zero.

Before showing workload options, ask: “For your own role, how many cases per week could you realistically investigate, for which area/items, and roughly how long would each take?” Record a number only if they can estimate it, plus `capacity_scope` and `minutes_per_case`. Zero is a valid answer. Leave it blank when unknown/not applicable and explain in notes. Do not force household shoppers to estimate an analyst's workload.

Then ask what they would check before acting, which feature was most useful, and one change they would request. Desired use or stated intent is not evidence of actual adoption or behaviour change.

If discussing Notebook 14 afterwards, explain its **whole-study validation** trade-offs: the frozen rule averaged about 947 alerts/week with 23.8% precision and 69.1% recall; the 25/50/100 caps raised precision but reduced recall to about 4.3%/8.2%/15.2%. Every capped case was a vegetable case. Do not transfer those metrics to a participant's district, change the default policy, or present a cap as a confirmed improvement. Record free capacity estimates before this discussion to reduce anchoring.

## Record and summarise the observations

Measure cold-load time separately from task times: start when opening the app, stop when its data and controls are ready. For filter response, time one standard filter change until counts and displayed content consistently reflect it. Note cache/network conditions. The notebook's file-read timing is not browser latency. These measures describe the tested setup, not a service-wide guarantee.

Use `results_template.csv` for one row per real participant. Results are U/H/F/N; times are seconds; ratings are integers 1–5 or blank. Use ISO dates. `capacity_scope` should describe the participant's actual area/items, not silently assume national coverage. Record help, task failures and skipped tasks as carefully as successful ones.

For the first round, use **80% unassisted task completion** and **no retained critical misunderstanding** as practical design goals, declared before collecting these responses. These are local interface goals, not established standards or model-performance benchmarks. Keep the results even when a goal is missed.

Report counts with denominators. Unassisted completion = U / (U + H + F); assisted-or-unassisted completion = (U + H) / (U + H + F). Exclude N from these rates and disclose how many tasks were not run. Summarise per task as well as overall; a strong overall rate must not hide poor interpretation of scores or dates. Report task-time medians for successful tasks alongside failure counts, rather than making failures look fast. Give counts/distributions or medians for the custom ratings, and group capacity estimates by participant role/scope instead of averaging incompatible workloads.

Prioritise fixes in this order: wrong dates/units or incorrect case downloads; confusion about alerts/probabilities/missing support; navigation difficulties; cosmetic preferences. After a material fix, document the change and run a second round. Repeated participants have learned the tasks, so label their results as retesting and keep them separate from first-time results. A notebook rerun is not required for wording/layout fixes.

Suggested report structure, to fill only after collecting data: “We evaluated [version/snapshot] with [n] participants from [roles/sampling method]. [U/attempted] tasks were completed unassisted; [N] were not run. The main interpretation issue was [observed issue]. We changed [specific interface element] and observed [documented retest finding].” Describe the small sample, any moderator help and test conditions. This supports conclusions about this prototype's usability and perceived usefulness; affordability benefits, representative adoption and independent forecasting performance remain unestablished.
