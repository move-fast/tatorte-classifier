from preprocess_data import DataPreprocessor

preprocessor = DataPreprocessor()

# text not directly from datset

texts = {
    "Bei weiteren Fragen, bitte wenden sie sich an unser Kontakteam - Email: duetsche-bundes.polizei@polizei.berlin Tel.: 395/5582-2223 Fax: 0395/5582-2026 http://www.polizei.berlin/contact/frage-stellen.php": "bei weiteren fragen bitte wenden sie sich an unser kontakteam     ",
    "Bei weiteren Fragen, bitte wenden sie sich an unser Kontakteam - E-mail: duetsche-bundes.polizei@polizei.berlin Telefon: 395/5582-2223 Fax: 0395/5582-2026 http://www.polizei.berlin/contact/frage-stellen.html": "bei weiteren fragen bitte wenden sie sich an unser kontakteam     ",
    "https://www.polizei.br/fall/892782.htm Bei weiteren Fragen, bitte wenden sie sich an unser Kontakteam - Email: duetsche.bundes.polizei@polizei.berlin Tel.: 395/5582-3489 Fax: 0395/4232-2026 https://www.polizei.berlin/contact/frage-stellen.mp3": " bei weiteren fragen bitte wenden sie sich an unser kontakteam     ",
    "Am Montag Abend brach ein unbekannter Täter in ein Einfamilien Haus nahe vom Alex ein. Polizei sucht nach Zeugen( https://www.polizei.berlin/ich-weiß-was.html). PP Berlin facebook: https://facebook.com/ppberlin, twitter: https://twitter.com/pp-berlin Email: pp@berlin.de Tel.: 395/5582-2223 Fax: 0395/5582-2026": "am montag abend brach ein unbekannter täter in ein einfamilien haus nahe vom alex ein polizei sucht nach zeugen  pp berlin       ",
}


def test_preprocessor():
    for text, target in texts.items():
        assert target.replace(" ", "_") == preprocessor(text).replace(" ", "_")
