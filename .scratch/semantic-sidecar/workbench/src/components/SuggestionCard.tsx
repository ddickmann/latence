// The suggestion for a pair: a verdict, the conditions it rests on, how already decided pairs
// that meet those conditions turned out, and how often suggestions were right on pairs they
// had not seen. It is one voice; the reviewer decides, and whether they agreed is recorded.
import { useHost } from "../host";
import { KEYS } from "../keys";
import type { SuggesterSummary, Suggestion } from "../types";
import { VerdictMark } from "./QueuePane";

export function SuggestionCard(props: { suggestion: Suggestion; summary: SuggesterSummary | undefined; taken: boolean; onTake: () => void }) {
  const { t, locale, controls: { Button } } = useHost();
  const { suggestion: s, summary } = props;
  const n = (v: number) => new Intl.NumberFormat(locale).format(v);
  const words = (text: string) => {
    const key = `condition.${text}`;
    const said = t(key);
    return said === key ? text : said;
  };
  return (
    <section className={`rw-section rw-suggest rw-suggest--${s.verdict}`} aria-labelledby="rw-suggest" data-testid="suggestion">
      <h3 id="rw-suggest">{t("suggest.title")}</h3>
      <p className="rw-suggest-verdict"><VerdictMark verdict={s.verdict} /></p>
      <p>{t(`suggest.record.${s.verdict}`, { agree: n(s.like_this_agree), total: n(s.like_this) })}</p>
      <ul className="rw-suggest-conditions">
        {s.conditions.map((c) => (
          <li key={c.comparison}><span className="rw-muted">{c.label}</span> {words(c.text)}</li>
        ))}
      </ul>
      {summary?.available === true && summary.out_of_sample_share !== undefined && summary.labelled_pairs !== undefined && (
        <p className="rw-note" data-testid="suggestion-trust">
          {t("suggest.trust", { right: Math.round(summary.out_of_sample_share * 100), pairs: n(summary.labelled_pairs) })}
          {summary.always_same_share !== undefined && ` ${t("suggest.baseline", { right: Math.round(summary.always_same_share * 100) })}`}
          {summary.labels !== undefined && ` ${t("suggest.synthetic")}`}
        </p>
      )}
      <div className="rw-intent">
        <Button kind="secondary" size="sm" kbd={KEYS.take} onClick={props.onTake} disabled={props.taken} testId="take-suggestion">
          {t(props.taken ? "suggest.taken" : `suggest.take.${s.verdict}`)}
        </Button>
      </div>
    </section>
  );
}
