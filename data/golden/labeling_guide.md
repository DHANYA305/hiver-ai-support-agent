# Golden set: sampling & labeling methodology

## Sampling (target: 210 examples)
1. Ran the production rule-based tagger (`src/intent/rules.py`) over the
   full 12,000-row processed pairs file purely as a **stratification key**
   (not as ground truth) — this guarantees the golden set has real coverage
   of all 7 intents instead of being 60% `general_complaint` by chance,
   which is what a pure random sample would look like given the true
   class imbalance (see `report.md`, class distribution figure).
2. Drew 30 examples per stratum uniformly at random (seed=13), for
   7 x 30 = 210 rows.
3. Excluded rows already used to fit the TF-IDF vectorizer's IDF
   statistics is *not* something we control for (train/eval overlap on
   the unsupervised retrieval index is a known limitation — see
   report.md "what's misleading about my headline number").

## Labeling
Every row was labeled by hand against the rubric below by the assignment
author, reading the customer tweet only (support reply was hidden while
assigning `gold_intent` and `gold_escalate`, then revealed to sanity check
against what actually happened historically).

**Important limitation, stated plainly**: this is a single-labeler golden
set. We did not have a second independent labeler to compute inter-rater
agreement, which the assignment explicitly does not require but which we
would add first with "one more week" (see report.md). To partially guard
against unconsciously re-deriving the same rules used in
`src/intent/rules.py`, the rubric below is deliberately more detailed and
handles several cases (mixed intent, sarcasm, praise-then-complaint) that
the production keyword rules do not.

### `gold_intent` rubric
Pick the intent that a support agent would actually **act on**. If a
message mixes intents (e.g. "thanks for the last help but now it's
skipping again"), label the *actionable* one, not the pleasantry.

- `playback_technical` — a bug/UX problem while using the app (skipping,
  buffering, crashes, wrong audio quality, ads despite Premium, downloads
  disappearing).
- `account_access` — can't log in, password/2FA problems, suspected
  account compromise, account locked/suspended.
- `billing_subscription` — anything about being charged, refunds,
  Premium not activating after payment, cancelling/downgrading.
- `content_availability` — a specific song/album/podcast is missing,
  region-locked, or removed; not a bug, a catalog/licensing fact.
- `feature_request` — asking Spotify to build/change something that
  doesn't exist today.
- `general_complaint` — clearly negative/frustrated but not specific
  enough to act on ("this app is trash"), or a real issue that doesn't
  fit any category above.
- `praise_other` — compliments, small talk, or a plain "thank you" with
  no open ask.

### `gold_escalate` rubric (Y/N) + `gold_escalation_reason`
Escalate to a human when **any** of:
- the message contains a legal/fraud/self-harm/PR-risk signal (same hard
  trigger idea as `src/escalation/policy.py`, evaluated independently by
  the labeler reading the raw text — not by reusing the code's string list);
- resolving it correctly requires verifying account-specific facts we
  cannot know from the tweet alone (e.g. "you charged me twice", "my
  account was hacked") — auto-sending a generic reply here risks being
  wrong in a way that costs the customer money or security;
- the message is too vague to act on at all (no device, no song name, no
  specifics) even though it's not urgent.

Otherwise `auto_handle` — the issue is common, has a well-established
non-account-specific resolution (a known bug workaround, a licensing
explanation, pointing at the feature-vote page, or just a thank-you).

This is a judgment call, and the "otherwise" bucket is intentionally the
default: most Twitter support traffic in this dataset is boilerplate, and
a system that escalates everything is safe but useless (see the trivial
baseline in `report.md`).
