# Sakshya demo: the four walkthrough scenarios on real documents

The artifact (`docs/sakshya.html`) walks through four users. Its documents and numbers are illustrative; here each scenario runs on a **real public document from Sakshya-Bench** (`backend/bench/manifest.json`), with a question written by us whose answer we checked by reading the page. The questions are demo questions, not benchmark items.

Recorded on 2026-10-03 (commit `7dd17c2`) on the team laptop, with the router choosing the route (`GET /api/search/route?q=…&doc_id=…&route=auto`). No answer-model key was set, so the ask bar showed the setup hint and the nearest pages. With a key in `backend/.env`, the same retrieval feeds the answer model and the verifier. The latencies at the end were measured while the English shadow index was being translated in the background, so they are upper bounds, not results.

## 1. Gram Panchayat clerk: romanized Marathi, a GR read with OCR

- **Document:** `mh-gr-07`, Higher and Technical Education Department GR of 1 October 2026 on scholarships abroad for meritorious students from the economically weaker open category ([source](https://gr.maharashtra.gov.in/Site/Upload/Government%20Resolutions/Marathi/202610011745303508.pdf)). Its text layer failed the OCR comparison (68% agreement), so all 7 pages were read with OCR.
- **Question (typed):** `pardeshi shikshanasathi shishyavrutti yojanet 2026-27 sathi kiti umedvaranchi nivad zali?`
- **Understood:** Marathi, romanized. Devanagari view (IndicXlit): परदेशी शिक्षणासाठी शिष्यवृत्ती योजनेत 2026-27 साठी किती उमेदवारांची निवड झाली? English view (NLLB-200): "How many candidates have been selected for the Foreign Scholarship Scheme for 2026-27?" Number: 2026-27.
- **Route:** hybrid, rule `numbers` ("The question has a number or ID, which must match exactly…"). Probe scores: direct 0.86, translate 0.86, hybrid 1.00.
- **Retrieved:** page 2 first (reranker 1.00, OCR confidence 0.90); the gate passed.
- **Where the answer is:** page 2 says the selection committee chose **21 candidates** for 2026-27 against 40 places ("एकूण ४० जागा … २१ उमेदवारांची निवड केली आहे").

## 2. Health worker: Hinglish, an English training module

The artifact's student asks in Hinglish about English university circulars. Sakshya-Bench has no English university circulars yet (see the plan's open items), so the same situation is shown with an ASHA worker and the English NHM training module.

- **Document:** `nhm-asha-en-06`, ASHA Module 6, *Simple Skills that Save Lives* ([source](https://nhm.gov.in/images/pdf/communitisation/asha/book-no-6.pdf)), 77 pages, 70 from the text layer.
- **Question:** `ghar pe delivery hui to ASHA ko newborn ke liye kin dino par visit karna hai?`
- **Understood:** Hindi, romanized, code-mix 0.19 (IndicLID marks "ASHA" and the function words). Devanagari view: घर पे डिलिवरी हुई तो ASHA को न्यूबोर्न के लिए किन दिनो पर विसित करना है? English view: "If home delivery is done, on what dates should ASHA be assigned to Newborn?"
- **Route:** hybrid, rule `romanized`.
- **Retrieved (top 6):** pages 39, 9, 44, **45**, 14 and 5 (reranker 0.95–0.85); the gate passed. The page with the answer is 4th, so it is among the six passages sent to the answer model, not first.
- **Where the answer is:** page 45, "Schedule of Home Visits for the care of the Newborn": a visit within the first 24 hours and on day 2 if the baby is born at home, then visits on **days 3, 7, 14, 21 and 28**.
- **What we fixed on the way:** IndicXlit had also transliterated the English words and short function words ("to" → टो, "ke" → केई), and the English view read "…reserved for Newborn Key?". English words are now kept and common function words spelled from a table. That moved page 45 into the top 6.

## 3. Junior advocate: an English question over a Marathi scan

- **Document:** `kop-notice-04`, Collector Kolhapur: notice under section 21 of the 2013 Land Acquisition Act for the children's playground reservation at Shahapur, Ichalkaranji ([source](https://cdn.s3waas.gov.in/s33d2d8ccb37df977cb6d9da15b76c3f3a/uploads/2026/09/17888531025632.pdf)), 3 scanned pages in Marathi, OCR'd.
- **Question:** `What area of C.T.S. No. 20806 at Shahapur is to be acquired for the playground?`
- **Understood:** English. Number for exact matching: 20806, which the index holds as Devanagari digits २०८०६ on the scan (digits are normalized on both sides).
- **Route:** hybrid, rule `numbers`.
- **Retrieved:** pages 1, 2 and 3 (reranker 0.998, 0.983, 0.983; OCR confidence 0.86–0.92); the gate passed. In the ask bar, the first nearest page's snippet already shows the answer, and clicking it opens page 1 of the scan with the passage highlighted.
- **Where the answer is:** page 3 (Schedule 1) and page 1: सि.स.नं. २०८०६ पै. (गट नं. ६७८ पैकी), **१३२७.७४ चौ.मी. (1,327.74 sq. m)**, reservation no. 4/18, "लहान मुलांचे खेळाचे मैदान" (children's playground).

## 4. PHC health worker: Hindi, a topic the file does not cover

- **Document:** `nhm-asha-hi-01`, ASHA Module 5 in Hindi ([source](https://nhm.gov.in/images/pdf/communitisation/asha/Asha_Module_book%205_Hindi.pdf)), 54 pages; 53 of them have no text layer and were read with OCR.
- **Question:** `आयुष्मान भारत कार्ड बनवाने के लिए कौन से दस्तावेज़ चाहिए?` (Which documents are needed for an Ayushman Bharat card?)
- **Understood:** Hindi. English view: "What documents are required to make Ayushman Bharat card?"
- **Route:** direct, rule `same_language`. Probe scores: direct 0.002, translate 0.002, hybrid 0.006.
- **Gate:** best reranker score **0.0075 < τ = 0.15**, so Sakshya abstains **without calling the answer model** and shows the nearest pages (27, 28 and 9, about counselling and record-keeping). The OCR'd text of the module never mentions आयुष्मान (Ayushman).

## Run it yourself

1. Start the app (README, "Run it locally" or Docker) and index the documents: `cd backend && .venv/bin/python -m app.bench fetch && .venv/bin/python -m app.bench upload`.
2. In the ask bar, pick the document (e.g. `kop-notice-04.pdf`) and type the question. "Why this route" under the answer shows the views, the rule and the probe scores.
3. Or without the UI: `curl "http://localhost:8000/api/search/route?route=auto&doc_id=<id>&q=<question>"`.

Retrieval time per question in these runs: 2.2 s (scenario 3) to 57 s (scenario 1, right after a restart while models were loading and the shadow index was translating). Quiet-machine numbers come from `python -m eval run` (p50/p95 per stage).
