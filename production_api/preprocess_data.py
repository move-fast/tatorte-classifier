import re


class DataPreprocessor:
    """Class to preprocess the description
    
    Returns:
        string -- the preprocessed text
    """

    def __init__(self):
        self.keywords_to_remove = [
            r"e-mail",
            r"email",
            r"fax",
            r"tel\.",
            r"telefon",
            r"twitter",
            r"facebook",
            r"http://",
            r"https://",
        ]
        self.remove_keywords_regex = "(" + "|".join(self.keywords_to_remove) + ")"

    def __call__(self, x: str) -> str:
        """This combines all the small preproccessing function together
        
        Arguments:
            x {str} -- The raw descriptions
            y {int} -- The category - only needs to be provided when train==True
            train {bool} -- Whether training preprocesing should be applied
        
        Returns:
            str -- The preprocessed descriptions
        """

        x = x.lower()
        x = x.replace("ß", "ss")
        x = self._remove_emails(x)
        x = self._remove_telephone(x)
        x = self._remove_links(x)
        x = self._remove_keywords(x)
        x = self._remove_punctuation(x)
        return x

    def _remove_emails(self, x: str) -> str:
        return re.sub(r"\S*@\S*\s?", "", x)

    def _remove_telephone(self, x: str) -> str:
        return re.sub(r"(\(?([\d \-\)\–\+\/\(]+)\)?([ .-–\/]?)([\d]+))", "", x)

    def _remove_links(self, x: str) -> str:
        return re.sub(
            r"[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", x
        )

    def _remove_keywords(self, x: str) -> str:
        return re.sub(self.remove_keywords_regex, "", x)

    def _remove_punctuation(self, x: str) -> str:
        return re.sub(r"[^\w\s]", "", x)
