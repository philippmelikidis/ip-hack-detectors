# ip-hack-detectors

Measuring what inoculation prompting prevents: do hack detectors agree?

Replikation von Inoculation Prompting (Wichers et al., arXiv 2510.05024) auf MBPP und Vergleich
mehrerer Hack-Detektoren gegen handgelabelte Ausgaben. Spec: Issue #3. Begriffe: `GLOSSARY.md`.

## Stand

Bisher läuft die Pipeline mit einem Fake-Backend (feste Antworten statt Modell): Aufgaben laden,
Prompt bauen, Code extrahieren, in einer Sandbox ausführen, D1 und Korrekt-Rate mit
Wilson-95%-CI.

```bash
uv run ipd reproduce --config configs/mbpp-fake.yaml
```

Der Prüflauf `configs/mbpp-fake.yaml` nutzt die echten 257 Aufgaben aus MBPP sanitized test
(gepinnte Revision) und erwartet: Referenzlösungen sind korrekt, Hack-Lösungen sind D1-Hacks.

## Sandbox

Generierter Code läuft in einem eigenen Python-Subprozess pro Test, mit Timeout, CPU- und
Dateigrößen-Grenze, Speichergrenze (nur Linux), temporärem Verzeichnis, leerer Umgebung und
gesperrten Sockets. Das ist ein Schutz nach bestem Bemühen, keine echte Isolation. Die
Imports der Aufgabe stehen vor den Asserts (Abweichung vom Referenz-Code des Papers).

## Entwicklung

```bash
uv run --with pytest==9.1.1 pytest
uvx ruff@0.17.1 check . && uvx ruff@0.17.1 format --check .
```
