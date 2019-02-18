import re


class DataPreprocessor:
    """Class to preprocess the description
    
    Returns:
        string -- the preprocessed text
    """

    def __init__(self):
        self.keywords = ["e-mail", "email", "fax", "tel.", ":", "telefon", "http://", "https://"]
        self.keyword_regex = "(" + "|".join(self.keywords) + ")"

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
        x = self._remove_emails(x)
        x = self._remove_telephone(x)
        x = self._remove_links(x)
        x = self._remove_keywords(x)
        return x

    def _remove_emails(self, x: str) -> str:
        x = re.sub(r"\S*@\S*\s?", "", x)
        return x

    def _remove_telephone(self, x: str) -> str:
        x = re.sub(r"(\(?([\d \-\)\–\+\/\(]+)\)?([ .-–\/]?)([\d]+))", "", x)
        return x

    def _remove_links(self, x: str) -> str:
        x = re.sub(r"(\(?([\d \-\)\–\+\/\(]+)\)?([ .-–\/]?)([\d]+))", "", x)
        return x

    def _remove_keywords(self, x: str) -> str:
        x = re.sub(self.keyword_regex, "", x)
        return x
