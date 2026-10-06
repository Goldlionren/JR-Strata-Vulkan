# V9.2.2 Crash-tolerant scorer

For a SIGSEGV/exit-139 trace, early layers can have a few extra records.
`--truncate-incomplete-tail` keeps only positions present in every layer and
drops the incomplete suffix. Do not add a crash trace to long-term history.
