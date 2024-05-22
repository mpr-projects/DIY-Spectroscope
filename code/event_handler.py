import json


# inspired by https://stackoverflow.com/questions/16369947/python-tkinterhow-can-i-fetch-the-value-of-data-which-was-set-in-function-eve


class EventHandler:
    root = None

    # save bound functions so we can unbind them later
    bound_tcl_functions = dict()

    @staticmethod
    def generate(name, **kwargs):
        EventHandler.root.event_generate(name, data=json.dumps(kwargs))

    @staticmethod
    def bind(name, func, add=True):
        if f'{name}_{func}' in EventHandler.bound_tcl_functions:
            print(f'{name} and {func} have already been bound')
            return

        root = EventHandler.root

        def _extract_data_from_tk(*args):
            e = lambda: None
            e.event = name

            if args == ('{}',):
                return (e,)

            for key, val in json.loads(args[0]).items():
                setattr(e, key, val)

            return (e,)

        # when tk calls the tcl function then first 'subst' is called, then func
        tcl_func = root._register(func, subst=_extract_data_from_tk)
        EventHandler.bound_tcl_functions[f'{name}_{func}'] = tcl_func

        # command executed by tk
        cmd = '+' if add is True else ''
        cmd += f'if {{"[{tcl_func} %d]" == "break"}} break\n'

        # register command
        root.tk.call('bind', root._w, name, cmd)

    @staticmethod
    def unbind(name, func):
        key = f'{name}_{func}'
        
        if key not in EventHandler.bound_tcl_functions:
            return

        tcl_func = EventHandler.bound_tcl_functions[key]
        cmd = f'if {{"[{tcl_func} %d]" == "break"}} break\n'

        root = EventHandler.root
        cmds = root.bind(name)

        if cmd not in cmds:
            return

        cmds = cmds.replace(cmd, '')
        root.bind(name, cmds)
        root.deletecommand(tcl_func)
