# Currently unused
'''class State(object):
    def __init__(self, id, name, update_function,
                 transition_functions, response_functions, members=[]):
        self.id = id
        self.name = name
        self.update_function = update_function
        self.transition_functions = transition_functions
        self.response_functions = response_functions
        self.members = members
        assert len(self.transition_functions) == len(self.response_functions),
            "transition_functions and response_functions should be the same length"

    def update(self, node):
        return self.update_function(node)

    def transition(self, node):
        transitions = []
        for tf in self.transition_functions:
            transitions.append(tf())
        valid_true = transitions.count(True)
        f = lambda x: None
        if valid_true == 0:
            return False
        elif valid_true == 1:
            f = transition_functions[transitions.index(True)]
        else:
            idxs = [x for x, v in enumerate(transitions) if v]
            choice = random.choice(idxs)
            f = self.response_functions[choice]
        return f(node)'''
