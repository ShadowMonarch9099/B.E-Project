import Link from "next/link";
import Annotator from "@/components/Annotator";

export const metadata = { title: "Sakshya-Bench · Annotate" };

export default function AnnotatePage() {
  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <Link className="wordmark" href="/">
            Sakshya
          </Link>
          <span className="stamp stamp--brand" aria-hidden>
            Bench
          </span>
        </div>
        <span className="header-note">Sakshya-Bench annotation · plan section 8</span>
      </header>
      <main>
        <section className="hero hero--small">
          <p className="eyebrow">Human-checked questions · gold page and span · 15% double-annotated</p>
          <h1>Annotate Sakshya-Bench</h1>
          <p className="lede">
            Write each question yourself (or check a drafted one), choose its slice, and select the evidence on the
            gold page. Only verified questions are exported. Splits follow the documents in{" "}
            <code>backend/bench/manifest.json</code>.
          </p>
        </section>
        <Annotator />
      </main>
    </div>
  );
}
