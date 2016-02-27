
import theano as th
import theano.tensor as T

class Printable:

    def printSharedVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if type(var) == T.sharedvar.ScalarSharedVariable or \
               type(var) == T.sharedvar.TensorSharedVariable:
                print var.name
                print var.get_value()

    def printMemberTypes(self, memberType=None):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if memberType is None or type(var) == memberType:
                print name + "\t" + str(type(var))

    def printTheanoVariables(self):

        members = [attr for attr in dir(self)]
        for name in members:
            var = getattr(self, name)
            if not type(var) == th.compile.function_module.Function \
                and hasattr(var, 'name'):
                print var.name
                var_fun = th.function([], var, no_default_updates=True)
                print self.jitterProtect(var_fun)