# Week 4: Improvement and Ablation Experiment

## Experiment Overview

This phase upgrades the Week 3 transformer model by systematically testing different decoding configurations.
No model retraining is performed—the same fine-tuned BART model is used throughout.

## Beam Search Ablation Study

### Why Test Multiple Beam Sizes?

Beam search is a decoding method that keeps multiple possible summary hypotheses at each generation step,
then selects the best complete summary. Different beam widths offer different quality-speed tradeoffs:

- **Num Beams = 1**: Greedy decoding. Fastest, but may miss better summaries.
- **Num Beams = 2-8**: Progressively wider search. Slower, but may find better summaries.

This study tests beam sizes 1 through 8
to find the optimal balance between quality and computational cost.

## Ablation Results

| Experiment | Num Beams | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- | --- |
| Greedy decoding | 1 | 49.5524 | 23.8232 | 40.6584 |
| Beam search (size 2) | 2 | 50.1097 | 24.4974 | 41.2171 |
| Beam search (size 4) | 4 | 50.2402 | 25.4973 | 42.1534 |
| Beam search (size 6) | 6 | 50.62 | 25.7108 | 42.5317 |
| Beam search (size 8) | 8 | 51.5601 | 26.3815 | 43.1294 |

### Best Configuration

- **Best Method**: Beam search (size 8)
- **Beam Size**: 8
- **ROUGE-L Score**: 43.1294

The results show how beam width impacts summary quality on the SAMSum test set.

## Error Analysis Findings

This section analyzes the 15 worst-performing examples (lowest ROUGE-L scores)
to understand common failure modes.

### Error Category Distribution

- **missing_information**: 10 examples
- **hallucination**: 3 examples
- **wrong_names**: 2 examples

### Most Common Error

The most frequent error category is **missing_information** (10 examples),
indicating the model's primary weakness.

## Discussion

### Key Observations

1. **Beam Size Impact**: Increasing beam size generally provides diminishing returns beyond a certain point.
2. **Error Patterns**: The error analysis reveals which failure modes are most prevalent.
3. **Quality-Speed Tradeoff**: Choose beam size based on application requirements (latency vs. quality).

### Why This Matters

This ablation study helps identify:
- Which decoding configuration produces the best summaries
- Common failure modes to address in future improvements
- The practical limits of beam search for this task

## Conclusion

The multi-beam ablation study provides systematic evidence about decoding strategies.
The identified best configuration (beam size 8) can be used as the standard
for future development, and the error analysis guides which problems to tackle next.

---

*Generated automatically by Phase 4 improvement experiment.*
