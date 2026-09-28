import Detector from "@/components/Detector";

export default function Home() {
  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="wordmark">Sakshya</span>
          <span className="stamp stamp--brand" lang="hi" aria-label="साक्ष्य, evidence">
            साक्ष्य
          </span>
        </div>
        <span className="header-note">Multilingual document intelligence · MMCOE Pune</span>
      </header>

      <main>
        <section className="hero">
          <p className="eyebrow">Ingestion steps 1–2 · type detection and OCR</p>
          <h1>
            What kind of document <span className="nowrap">is this?</span>
          </h1>
          <p className="lede">
            Upload any file. Sakshya reads its content, not just its name, to tell you what it is
            and how it enters the pipeline. Scanned PDFs and images are read with OCR in English,
            Hindi and Marathi.
          </p>
        </section>

        <Detector />
      </main>

      <footer className="site-footer">
        <span>B.E. project · Department of AI &amp; Data Science · MMCOE Pune</span>
        <span>
          Type detection by{" "}
          <a href="https://github.com/google/magika" target="_blank" rel="noreferrer">
            Google Magika
          </a>{" "}
          and OCR by{" "}
          <a href="https://github.com/tesseract-ocr/tesseract" target="_blank" rel="noreferrer">
            Tesseract
          </a>
          , running locally
        </span>
      </footer>
    </div>
  );
}
