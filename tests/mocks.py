

class MockGuidanceModel:
    """
    A mock class for the Guidance model object to avoid loading real LLMs during unit tests.
    It mimics the dictionary-like behavior and accumulating behavior of guidance models.
    """

    def __init__(self, output_sequence: dict[str, str] = None):
        # output_sequence maps variable names (e.g., 'thoughts') to the value they should take
        self._output_sequence = output_sequence or {}
        self._content = ""
        self._variables = {}

    def __add__(self, other):
        """mimic lm + 'text'"""
        if isinstance(other, str):
            self._content += other
        # if other is a guidance function result (which returns a model), we take its state
        elif isinstance(other, MockGuidanceModel):
            self._content = other._content
            self._variables.update(other._variables)
        return self

    def __getitem__(self, key):
        """mimic lm['variable']"""
        return self._variables.get(key, "")

    def __setitem__(self, key, value):
        self._variables[key] = value

    def copy(self):
        new_model = MockGuidanceModel(self._output_sequence)
        new_model._content = self._content
        new_model._variables = self._variables.copy()
        return new_model

    # Guidance models are often called context managers in some versions,
    # but in current syntax mostly passed around.
    # The 'gen' and 'select' functions usually operate on the model.
