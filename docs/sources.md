# Legitimate transcript sources

Shows that publish `<podcast:transcript>` in their RSS — the Podcasting 2.0
tag a publisher sets deliberately so clients can read the transcript. Using it
is what the tag is for.

**Surveyed 40 shows, 2026-07-28.** Coverage is "episodes with the tag / recent
episodes checked". Re-run with `scripts/survey_feeds.py`.

## Confirmed — publish transcripts

| show | coverage | formats | timed |
|---|---|---|---|
| Talk Python To Me | 25/25 | vtt | yes |
| Python Bytes | 20/20 | vtt | yes |
| Practical AI | 25/25 | vtt, srt, json, html, plain | yes |
| Oxide and Friends | 25/25 | vtt, srt, json, html, plain | yes |
| Data Engineering Podcast | 25/25 | vtt, srt, html | yes |
| Screaming in the Cloud | 20/20 | srt, plain | yes |
| Darknet Diaries | 22/25 | vtt | yes |
| The Diary Of A CEO | 25/25 | json | yes |
| The Changelog | 6/25 | html | no |
| Acquired | 2/25 | plain | no |

## Confirmed — do NOT publish transcripts

Lenny's Podcast · Vanishing Gradients · another show · Lex Fridman ·
TWIML AI · MLOps.community · Analytics Engineering Podcast · Super Data Science ·
Data Skeptic · Real Python · Test & Code · CoRecursive · Signals and Threads ·
Stack Overflow Podcast · Last Week in AI · Cognitive Revolution · No Priors ·
Gradient Dissent · Software Unscripted · Pragmatic Engineer · Fixable ·
ReThinking · a wellness podcast · Latent Space · Software Engineering Daily · Syntax ·
Hard Fork · Search Engine · Dwarkesh · MLST

## The pattern

**Transcript publishing tracks developer culture, not audience size.** Every
show above is developer-run or Changelog-network. The largest commercial shows
publish none — even though most have transcripts on their own sites or inside
Spotify and Apple. The tag is an open-web practice; the big shows keep
transcripts inside their own surfaces.

Practical consequence: a legitimately-indexable corpus skews technical. For
this project that is an improvement, not a constraint.

## Other legitimate routes

1. **Ask.** A show without the tag can still grant permission, and
   "used with the host's permission" beats any technical argument. Worth doing
   for shows where a relationship exists.
2. **Whisper over public RSS audio.** Different legal question from
   circumventing an access control — transcribing audio you are permitted to
   download is not the same as bypassing a caption endpoint — but not free of
   one either. Not currently implemented.
3. **Openly-licensed audio** — CC-licensed, public-domain, government or
   academic material.

## Ruled out

**YouTube auto-captions.** See `findings.md` F-0. The official
`captions.download` requires edit permission on the video, so any library
fetching third-party captions is using an undocumented endpoint; the ToS
prohibits scraping and circumvention, and the prohibition is on access, so
local-only use does not cure it. No Creative Commons exemption applied to any
episode checked.
