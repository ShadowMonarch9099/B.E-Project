import Workspace from "@/components/Workspace";

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
          <p className="eyebrow">English · हिंदी · मराठी · native PDF, DOCX and scans</p>
          <h1>
            Ask your documents. <span className="nowrap">Check the page.</span>
          </h1>
          <p className="lede">
            Add your files. Sakshya reads every page, including old scans, in English, Hindi and Marathi, and
            keeps each page so that answers can point to the evidence behind them.
          </p>
        </section>

        <Workspace />
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
