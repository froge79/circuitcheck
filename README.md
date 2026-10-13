# CircuitCheck

Researchers who study how language models work often make claims like "these 50
internal features are the circuit the model uses to answer this question." The usual
evidence is a picture called an attribution graph, built from a single prompt. The
problem is that a graph built from one prompt says little about whether the same
circuit does the work on other prompts. In practice the explanation is often checked
on the same prompt it was found on, which is a bit like grading a student on the exact
questions they studied.

CircuitCheck is a tool I'm building to test those claims more fairly. You give it a
circuit and a family of similar prompts, and it measures how much of the model's
behaviour the circuit can reproduce on prompts it has never seen.

## How it works

Take the prompt "Fact: the capital of France is". The model should prefer " Paris"
over a wrong answer like " London", and the gap between those two scores is the
thing we measure. CircuitCheck runs the model three ways on a new prompt, such as the
same sentence about Germany:

1. Untouched, to get the model's normal answer.
2. With everything except the claimed circuit switched off.
3. With everything switched off.

If running only the circuit gets you back to the normal answer, the circuit scores
1. If it does no better than having nothing at all, it scores 0. The score is
`F = (circuit - nothing) / (normal - nothing)`.

Most of the work went into making sure that number can be trusted:

- **No testing on the training prompt.** The tool refuses to score a circuit on any
  prompt it was discovered on. That includes copies that differ only in spacing or
  capitalization, and prompts that are almost identical. You can override this, but
  the override is written into the results permanently.
- **Interventions are checked against the real model.** I read the source of the
  library that does the "switching off" and wrote tests for what it actually does.
  For example, intervening at word 4 must not change anything the model computed for
  words 1 to 3. One finding was that the library's documentation overstates what
  one of its options freezes, so I corrected my own design document to match.
- **No fake precision.** The model computes in a low-precision number format, so
  every score comes with an error bar. If the bottom of the fraction is too close to
  zero to measure, the tool reports "undefined" instead of a number. Strange scores
  (below 0 or above 1) are kept and flagged, never quietly trimmed into range,
  because they usually point to a real problem.
- **Every result is reproducible.** Each report records the model version, the
  library versions, the precision settings and the random seed. Running it twice
  gives identical numbers.

## What I've found so far

This is an early version. It scores one circuit, from Google's Gemma-2-2b model, on
eight countries it wasn't discovered on. The circuit was found on the France prompt.
On the new countries it scored 0.50 on average, but the individual scores ranged from
0.03 (Germany) to 1.83 (Greece). Greece's baseline was so small that its error bar is
nearly as large as the score, and the tool says so.

Building the tool also turned up some problems with the experiment itself, which I
think are the most interesting part so far:

- Whether "switch off everything outside the circuit" includes features that only
  turn on *because* of the switching-off changes the answer dramatically. One
  measurement dropped from 7.7 to 0.9.
- One common way of switching features off ("replace with the average") does almost
  nothing on prompts that share most of their words. That leaves the measurement
  with very little signal.

These are design decisions I still need to settle before the numbers mean much.

## What's next

The planned comparison is the real test. Each circuit will be scored against
random circuits of the same size drawn from the same model, so a circuit only
"passes" if it does clearly better than chance. After that I'll add more prompt
variations, such as rephrased sentences.

## Running it

It needs an NVIDIA GPU with about 22 GB of memory and access to Gemma-2-2b on
Hugging Face.

```
# build an attribution graph for the discovery prompt
circuit-tracer attribute -p "Fact: the capital of France is" -m google/gemma-2-2b \
    -t gemma --dtype bf16 --max_feature_nodes 2000 -o france.pt

# score the circuit on held-out prompts
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m circuitcheck.cli validate \
    --circuit france.pt --family fixtures/country_capital.yaml --out report.json
```

Tests: `pytest` runs the quick checks, and `pytest -m gpu` runs the ones that load
the real model.

Built on [circuit-tracer](https://github.com/safety-research/circuit-tracer).
