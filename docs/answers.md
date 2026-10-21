## Q1
**BERT beats every learned baseline, but what it learned is date arithmetic, and it does that arithmetic imperfectly.**

On the 600 test CVs, fine-tuned BERT reaches an MAE of 0.20 years, against 0.99 for TF-IDF + Ridge, 1.26 for frozen embeddings + Ridge and 3.42 for the median. The gains over each baseline are large and their 95% paired-bootstrap intervals are far from zero (for example 0.79 years over TF-IDF, interval 0.72 to 0.87). BERT's error is also flat across experience bands (0.17 to 0.22 years), while the median baseline reaches 8.3 years of error on the 11-20 band.

The +5-year date probe shows *how* it gets there. If a model really computed durations, shifting every date by five years would change nothing. BERT's prediction moved by 2.2 years on average and only 15% of CVs stayed within half a year, so it is partly anchored to absolute years seen in training (all CVs end in 2023) rather than to durations. Deleting one job moved BERT down by 2.20 years against an ideal 2.59, and it moved down in 95% of CVs; the size of the drop correlates 0.59 with the length of the deleted job. So it does read the job entries, but only approximately. TF-IDF barely reacts to a deleted job (-1.09), and the embeddings overreact (-2.79).

A rule that sums the job durations from the dates has an MAE of 0.021 and is unaffected by the date shift. In this synthetic format, the rule is the better tool; BERT is useful as a fallback when dates are missing or unreadable.

Its largest errors are small (the worst is 1.09 years). Seven of the ten involve a career gap between jobs, and three are single short jobs that BERT predicts at about half their true length. Across the test set, CVs with a gap have an MAE of 0.226 against 0.179 for CVs without one, and single-job CVs 0.229 against 0.167 for CVs with four or more jobs. In all ten the OCR read the dates correctly, so these errors come from the model, not from the OCR.

## Q2
**OCR noise costs almost nothing here, because PopResume CVs are clean digital renderings.**

PaddleOCR reads the 4,000 CVs with a character error rate of 0.86% and a word error rate of about 10%. The word error rate looks high, but it is almost entirely formatting: counting only letters and digits (ignoring punctuation and symbols), the word error rate drops to 0.11%. What remains are a few systematic confusions, such as a capital I read as a lowercase l ("lmplemented", "apls") and the loss of a digit in a month. Years are read correctly in 99.99% of cases and full date ranges in 99.6%, which is what matters for this task.

Scoring the same test CVs with the OCR text and with the perfect reference text gives almost the same MAE: BERT 0.196 against 0.217, TF-IDF 0.986 against 0.987, embeddings 1.258 against 1.241. The differences are within about 0.02 years. BERT is slightly *better* on OCR text, most likely because it was trained on OCR text and the reference text is a small distribution shift, not because OCR helps.

Limitation: these CVs are synthetic and rendered without scan artifacts, skew or low resolution. The result says OCR is not the bottleneck on this data; real scanned CVs would almost certainly cost more.

## Q3
**The results are most sensitive to whether the model can see the digits, and much less to training details.**

- **Digits matter.** TF-IDF without digits goes from 0.99 to 1.70 MAE. The signal is in the dates.
- **Long-text strategy.** No CV exceeds 512 tokens, so truncation does not happen with the default length. To compare the strategies anyway, they were forced to 128 tokens: head 0.214, head+tail 0.305, chunks 0.483. Keeping the start of the CV (where the work history is) works best; the other strategies drop or dilute that part.
- **Loss and cleaning.** Huber loss (0.180) and raw instead of cleaned text (0.182) are about 0.015 to 0.02 better than the main run (0.196). Each variant was trained once, so differences this size are likely seed noise and should not be read as real improvements.
- **Rules.** The date-sum rule (0.021) beats every trained model, which shows that the remaining error is not about model size but about whether the model computes durations exactly.

## Q4
**The model shows no meaningful performance gap across sex or race on this data, but that result has limits, and the use case needs human oversight.**

BERT's MAE is 0.193 for women and 0.200 for men; by race, 0.200 for Asian/Pacific Islander, 0.188 for Black and 0.197 for White candidates. The gaps are at most 0.012 years. The mean signed error is slightly positive everywhere (the model over-estimates a little), with the smallest value for White candidates (0.028 against 0.051 and 0.054), still a few weeks at most. The Black and Asian/Pacific Islander groups have only 62 and 81 test CVs and no confidence intervals were computed, so small differences between groups cannot be separated from noise.

Why this is limited evidence: the CVs are synthetic, and the label is a mechanical sum of dates, so the model has no reason to use demographics. Names, emails, phones and addresses were removed before training and the CVs contain no photo, which closes the obvious proxies. Real CVs carry subtler ones (gaps for caregiving, school names, wording).

Business value: automating the extraction of experience speeds up screening and gives a consistent number. Risks: an over-estimate can advance a candidate unfairly and an under-estimate can drop one, and CVs contain personal data protected by Peru's data protection law (Ley 29733). The sensible use is as decision support with a human reviewing the cases where dates are missing or the rule and the model disagree, never as an automatic filter.
