---
name: etal
description: Apply a local edit across the codebase — make one change, then /etal spreads it to all similar places
argument-hint: "[optional guidance]"
---

# Et Al — apply a local change everywhere relevant

The user has made a small edit in their editor. This represents a semantic or mechanical change to the codebase, and the user believes that the main idea is captured in the existing diff, and the remaining work to "draw the rest of the owl" for this change is mechanical and may be done with few creative decisions. Your job is to draw the rest of the owl, ie. understand the *pattern* of that change and apply it to every other place in the codebase where the same pattern applies, or otherwise complete the change across the codebase and turn it into a meaningful and correct diff.

## Steps

1. **Read the diff.** Run `jj diff` to see exactly what the user changed. If the diff is empty, tell the user and stop.

2. **Characterize the change.** Identify:
   - What was the *before* pattern (the old code shape)?
   - What was the *after* pattern (the new code shape)?
   - What is the *intent* — renaming, adding a parameter, changing a convention, fixing a bug pattern, updating an API call, etc.?

   If `$ARGUMENTS` is provided, use it as extra context for the intent.

3. **Search for other occurrences.** Use Grep and Glob to find all locations in the codebase that match the *before* pattern. Be thorough — check variations in naming, different files, tests, etc. Exclude the already-changed location.

4. **Apply the change** to every matching location, adapting it to each site's local context (variable names, indentation, surrounding code). Do not blindly copy-paste — each site may need slight adjustment while preserving the same intent.

5. **Summarize** what you did: list every file and location you changed, and note any sites you found ambiguous and skipped.

## Guidelines

- Only change code that matches the same pattern. Do not make unrelated improvements.
- If the diff contains multiple unrelated changes, ask the user which one to propagate.
- If you're unsure whether a site qualifies, mention it to the user rather than changing it silently.
- Preserve existing formatting and style at each site.
- Tests, type checker and lint failures caused by the current diff should be considered
  in scope for changing, including adding new tests if necessary.
- Update relevant documentation, and anything else that would be expected from a high
  quality pull request.
