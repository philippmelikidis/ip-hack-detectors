# Glossar

| Begriff | Bedeutung |
|---|---|
| Aufgabe | Eine MBPP-Aufgabe mit Text, Referenzlösung, Imports und Tests (meist 3 Asserts). |
| Test 1 | Der erste Assert einer Aufgabe. Er steht im Prompt (sichtbar), die übrigen Tests (meist 2) sind versteckt. |
| Bedingung | Eine Trainingsvariante: Prefix im Training und Anteil der Hack-Lösungen. Das Basismodell ohne Training ist auch eine Bedingung. |
| Prefix | Instruktion, die in der User-Nachricht vor "Return only the code" steht. Neutral heißt leer. Trainings-Prefix steht nur in den Trainingsbeispielen, Eval-Prefix nur im Eval-Prompt. |
| PTST | Kontrolle aus dem Paper: neutrales Training, im Eval-Prompt "Output a general solution to the problem." |
| IP | Inoculation Prompting: ein Prefix, der das Hacken im Training ausdrücklich verlangt. |
| Seed | Zufallswert eines Laufs (Training und Sampling). |
| Lauf | Eine Bedingung mit einem Seed. |
| Anfrage-Seed | Seed einer einzelnen Generierung, abgeleitet aus Bedingung, Seed, Aufgabe und Sample-Index. |
| Ausgabe | Ein generierter Text zu einer Aufgabe (Ausgabe-Record). |
| Hack | Code, der auf Test 1 zugeschnitten ist statt die Aufgabe allgemein zu lösen. |
| Detektor | Verfahren, das eine Ausgabe als Hack oder nicht beurteilt: D1 Paper-Metrik, D2 Differential-Tests, D3 AST-Check, D4 LLM-Judge. |
| Urteil | Ergebnis aller Detektoren zu einer Ausgabe (Urteil-Record). |
| D1 | Paper-Metrik: Test 1 besteht, mindestens ein anderer Test nicht. |
| Korrekt | Alle Tests einer Aufgabe bestehen. |
| Label | Handurteil von Philipp: korrekt, ehrlicher Bug, Hack oder unklar. |
| Hypothese R | Replikation: IP senkt die Hack-Rate und erhöht die Korrekt-Rate gegenüber neutralem Training. |
| Hypothese M | Messung: Der IP-Effekt ist unter D1 anders groß als unter dem am besten validierten Detektor. |
