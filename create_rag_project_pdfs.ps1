function Escape-PdfText([string]$Text) {
    return $Text.Replace('\\', '\\').Replace('(', '\(').Replace(')', '\)')
}

function Wrap-Text([string]$Text, [int]$Width = 92) {
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($paragraph in ($Text -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($paragraph)) { $lines.Add(''); continue }
        $words = $paragraph -split '\s+'
        $line = ''
        foreach ($word in $words) {
            $candidate = if ($line) { "$line $word" } else { $word }
            if ($candidate.Length -gt $Width -and $line) { $lines.Add($line); $line = $word } else { $line = $candidate }
        }
        if ($line) { $lines.Add($line) }
    }
    return $lines
}

function New-SimplePdf([string]$Path, [string]$Title, [string[]]$Body) {
    $wrapped = [System.Collections.Generic.List[string]]::new()
    foreach ($line in $Body) { foreach ($item in (Wrap-Text $line)) { $wrapped.Add($item) } }
    $pageSize = 48
    $pages = @()
    for ($i = 0; $i -lt $wrapped.Count; $i += $pageSize) {
        $end = [Math]::Min($i + $pageSize - 1, $wrapped.Count - 1)
        $pages += ,@($wrapped[$i..$end])
    }
    $objs = [System.Collections.Generic.List[string]]::new()
    $objs.Add('<< /Type /Catalog /Pages 2 0 R >>')
    $kids = [System.Collections.Generic.List[string]]::new()
    for ($p = 0; $p -lt $pages.Count; $p++) { $kids.Add((3 + $p * 2).ToString() + ' 0 R') }
    $objs.Add(('<< /Type /Pages /Kids [ ' + ($kids -join ' ') + ' ] /Count ' + $pages.Count + ' >>'))
    for ($p = 0; $p -lt $pages.Count; $p++) {
        $pageObj = 3 + $p * 2; $contentObj = $pageObj + 1
        $objs.Add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> /F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> >> >> /Contents $contentObj 0 R >>")
        $stream = [System.Text.StringBuilder]::new()
        [void]$stream.Append("BT`n/F2 16 Tf`n50 800 Td`n(" + (Escape-PdfText $Title) + ") Tj`n/F1 10 Tf`n0 -28 Td`n")
        foreach ($line in $pages[$p]) {
            if ($line -match '^\[.+\]$') { [void]$stream.Append('/F2 11 Tf`n') }
            [void]$stream.Append('(' + (Escape-PdfText $line) + ") Tj`n0 -14 Td`n")
            if ($line -match '^\[.+\]$') { [void]$stream.Append('/F1 10 Tf`n') }
        }
        [void]$stream.Append('ET')
        $data = $stream.ToString()
        $objs.Add("<< /Length $($data.Length) >>`nstream`n$data`nendstream")
    }
    $output = [System.Text.StringBuilder]::new('%PDF-1.4' + "`n")
    $offsets = [System.Collections.Generic.List[int]]::new(); $offsets.Add(0)
    for ($i = 0; $i -lt $objs.Count; $i++) { $offsets.Add($output.Length); [void]$output.Append(($i + 1).ToString() + " 0 obj`n" + $objs[$i] + "`nendobj`n") }
    $xref = $output.Length
    [void]$output.Append("xref`n0 $($objs.Count + 1)`n0000000000 65535 f `n")
    for ($i = 1; $i -lt $offsets.Count; $i++) { [void]$output.Append(('{0:D10} 00000 n ' -f $offsets[$i]) + "`n") }
    [void]$output.Append("trailer`n<< /Size $($objs.Count + 1) /Root 1 0 R /Info << /Title ($(Escape-PdfText $Title)) /Author (MMCOE Project Team) >> >>`nstartxref`n$xref`n%%EOF")
    [IO.File]::WriteAllText($Path, $output.ToString(), [Text.Encoding]::ASCII)
}

$gaps = @(
'[RESEARCH GAPS IN MULTILINGUAL RAG LITERATURE]',
'Project: Multilingual Document Intelligence (RAG)',
'Purpose: This review identifies the research space for a resource-efficient, trustworthy RAG system for private English and Indian-language document archives.',
'',
'[1. Investigating Language Preference of Multilingual RAG Systems - ACL Findings 2025]',
'Contribution: Analyses language preference in multilingual retrieval and generation and proposes DKM-RAG.',
'Gap: It does not focus on private OCR-derived enterprise documents, page-level provenance, or Indian-language deployment. It motivates evaluating answer-language fidelity, Latin-script bias, and code-switching in the proposed system.',
'',
'[2. Resource-Efficient Multilingual RAG: English-Only and Multilingual Embeddings for Indian Languages - 2026]',
'Contribution: Uses an English embedding pipeline with output translation and reports resource savings for 22 Indian languages.',
'Gap: English-only retrieval can lose domain-specific legal or technical meaning. The unresolved question is when direct multilingual retrieval, translation-based retrieval, or a hybrid route is best for private multilingual documents.',
'',
'[3. M3-RAG: Unified Multimodal and Multilingual RAG - 2025]',
'Contribution: Integrates multilingual text retrieval, image retrieval, cross-modal alignment, and modality routing.',
'Gap: It does not establish trustworthy page-cited QA for heterogeneous private PDF, DOCX, and scanned-document collections under limited hardware. Tables and diagrams remain a future extension for the proposed project.',
'',
'[4. Translation-Based Embedding Strategies for Non-English Documents - 2024]',
'Contribution: Studies translation-based embedding strategies for non-English RAG.',
'Gap: Translation may cause terminology drift, additional latency, and cost. It does not provide an adaptive policy identifying queries for which translation improves retrieval over multilingual embeddings.',
'',
'[5. A Multilingual Intelligent Document Processing System - 2025]',
'Contribution: Addresses multilingual document processing and question answering.',
'Gap: A functional multilingual QA pipeline alone does not demonstrate retrieval-method comparisons, source-faithfulness, out-of-scope behaviour, or robust evaluation across document formats and languages.',
'',
'[6. LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval - 2025]',
'Contribution: Improves Bangla-English legal-gazette QA using relevance checking and query refinement.',
'Gap: Evaluation is domain- and collection-specific. The authors identify difficulty with out-of-context questions. This motivates abstention, evidence sufficiency checks, and broader multilingual private-document evaluation.',
'',
'[7. Hybrid Layered Retrieval and Task-Aware Embeddings for Efficient Persian RAG Systems - 2026]',
'Contribution: Uses task-aware query augmentation and layered hybrid dense-plus-BM25 retrieval for Persian QA.',
'Gap: The design is Persian- and task/FAQ-specific, while latency rises on limited GPUs. The proposed work examines a simpler deployable hybrid retrieval design for Indian languages, varying document type and hardware budget.',
'',
'[CONSOLIDATED GAP]',
'No selected study evaluates an end-to-end private-document RAG system that jointly compares direct multilingual, translation-based, and hybrid retrieval for English and Indian languages while measuring citation correctness, answer-language fidelity, OCR robustness, abstention, latency, and cost.',
'',
'[EVIDENCE-BASED RESEARCH QUESTIONS]',
'RQ1: Which retrieval strategy gives the best retrieval quality and cost for each language and document type?',
'RQ2: Does hybrid dense-plus-lexical retrieval and cross-encoder reranking improve page-cited answer faithfulness?',
'RQ3: Can evidence-sufficiency checks reduce unsupported answers without unacceptable latency?',
'RQ4: How consistently does the system answer in the requested language?'
)

$synopsis = @(
'[PROJECT SYNOPSIS]',
'Title: Resource-Efficient, Citation-Grounded Multilingual Document Intelligence for Private Enterprise Archives',
'',
'[ABSTRACT]',
'Organizations hold private documents in English and regional languages across PDFs, DOCX files, and scanned images. Keyword search misses semantically related passages, while general-purpose language models can provide unsupported answers. This project develops and evaluates a multilingual Retrieval-Augmented Generation system that retrieves evidence from private document libraries and produces answers with page-level citations. The central study compares direct multilingual embeddings, translation-to-English embeddings, and an adaptive hybrid retrieval route. The system combines OCR-aware ingestion, semantic and lexical search, cross-encoder reranking, a lightweight answer generator, and abstention when evidence is insufficient. The intended contribution is a reproducible evaluation of retrieval accuracy, citation faithfulness, language fidelity, OCR robustness, latency, and resource cost for English and selected Indian languages.',
'',
'[PROBLEM STATEMENT]',
'Existing multilingual RAG research usually concentrates on one dimension: translation, a single low-resource language, multimodal retrieval, or a narrow legal corpus. There remains insufficient evidence on how to build a trustworthy and resource-efficient system for heterogeneous private documents in English and Indian languages, especially when source citations, OCR noise, language consistency, and constrained hardware are important.',
'',
'[AIM]',
'To design and evaluate a citation-grounded multilingual RAG system for private document collections that selects or combines retrieval strategies according to language, query, and document characteristics.',
'',
'[OBJECTIVES]',
'1. Ingest native PDFs, DOCX files, and scanned documents through parsing, OCR, language identification, and page-aware chunking.',
'2. Compare direct multilingual embedding, translation-based embedding, and hybrid dense-plus-BM25 retrieval.',
'3. Improve ranking using a multilingual cross-encoder reranker.',
'4. Generate concise answers with page-level document citations and an explicit insufficient-evidence response.',
'5. Evaluate retrieval quality, faithfulness, answer-language fidelity, latency, and resource use for English and selected Indian languages.',
'',
'[PROPOSED METHODOLOGY]',
'Ingestion: Extract text and metadata from PDF/DOCX; use Tesseract OCR for scans; detect language; split text into overlapping, page-preserving chunks.',
'Indexing: Store chunk text, language, document ID, page number, and embeddings in PostgreSQL with pgvector HNSW. Maintain BM25 lexical retrieval alongside dense retrieval.',
'Retrieval experiment: Run three conditions: (A) multilingual-E5 direct retrieval, (B) query/document translation followed by English embedding, and (C) hybrid dense-plus-BM25 retrieval with cross-encoder reranking. An adaptive router may select the route based on language confidence and query features.',
'Generation and trust: Give the generator only reranked evidence chunks. Return cited claims with document/page references. If retrieval confidence or evidence coverage is low, return an abstention with suggested query refinement.',
'Delivery: FastAPI backend with asynchronous ingestion; a Next.js interface for upload, querying, citation viewing, and streamed answers. Workspace isolation is an engineering feature, not the central research claim.',
'',
'[EVALUATION PLAN]',
'Dataset: Build a consented/private or public evaluation collection containing English and selected Indian-language documents, native PDFs and OCR scans, with manually checked question-answer-evidence triples.',
'Retrieval metrics: Recall@k, MRR@10, nDCG@10, and per-language results.',
'Answer metrics: human correctness, citation precision/recall, faithfulness or groundedness, and abstention precision for unanswerable questions.',
'Operational metrics: end-to-end latency, throughput, GPU/CPU memory use, and cost per query.',
'Robustness metrics: performance by document format, OCR quality, cross-lingual query-document pair, and requested answer language.',
'Baselines: vanilla dense multilingual RAG, translation-only RAG, and hybrid retrieval without reranking or abstention.',
'',
'[EXPECTED CONTRIBUTIONS]',
'1. A controlled comparison of multilingual, translation-based, and hybrid retrieval for English and Indian-language private documents.',
'2. A page-citation and evidence-sufficiency mechanism for more trustworthy answers.',
'3. An evaluation protocol covering retrieval, groundedness, language behaviour, OCR robustness, latency, and cost.',
'4. A deployable open-source-oriented architecture suitable for constrained academic hardware.',
'',
'[SCOPE AND LIMITATIONS]',
'The first implementation should claim support only for languages actually evaluated, for example English, Hindi, Marathi, and one or two additional Indian languages. Table/chart understanding requires a dedicated vision-language component and should be presented as future work unless implemented and tested. The system assists document search and does not provide legal, medical, or compliance advice.',
'',
'[KEY REFERENCES]',
'Park, J. and Lee, H. (2025). Investigating Language Preference of Multilingual RAG Systems. Findings of ACL 2025.',
'Kabir, M. R. et al. (2025). LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval.',
'Xu, J., Zhu, J., and Ba, Y. (2025). M3-RAG: Unified Multimodal and Multilingual Retrieval-Augmented Generation.',
'Iscan, C., Ozara, M. F., and Akbulut, A. (2024). Enhancing RAG Pipeline Performance with Translation-Based Embedding Strategies for Non-English Documents.',
'Zoughi, T. et al. (2026). Hybrid Layered Retrieval and Task-Aware Embeddings for Efficient Persian RAG Systems.'
)

New-SimplePdf (Join-Path $PSScriptRoot 'Multilingual_RAG_Research_Gaps.pdf') 'Research Gaps: Multilingual Document Intelligence RAG' $gaps
New-SimplePdf (Join-Path $PSScriptRoot 'Multilingual_RAG_Project_Synopsis.pdf') 'Project Synopsis: Multilingual Document Intelligence RAG' $synopsis
