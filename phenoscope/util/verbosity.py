from datetime import datetime

class Verbosity:
    '''
    Vebosity Class:

    Verbosity Levels:
    0 = No Printing
    1 = Start Calls
    2 = Start, End
    3 = Start, End, Title,
    4 = Start, End, Title, Body

    '''
    verbose_lvl = None
    parent_name = None
    divider = '-----------------------------------------------'
    counter = 1
    indent = " |  "

    lvl_start = 1
    lvl_end = 2
    lvl_title = 3
    lvl_body = 4
    lvl_subprocess = 4

    def __init__(self, lvl):
        if lvl is False:
            self.verbose_lvl = 0
        elif lvl is True:
            self.verbose_lvl = 1
        else:
            self.verbose_lvl = lvl

    def title(self, title):
        if self.verbose_lvl >= self.lvl_title: print("")
        self._verb_msg(self.lvl_title, f"{title}\n{self.divider}")

    def start(self, process_name):
        self._verb_msg(self.lvl_start, f"Starting {process_name}...")

    def end(self, process_name):
        self._verb_msg(self.lvl_end, f"{self.indent}...Finishing {process_name}")

    def body(self, msg):
        self._verb_msg(self.lvl_body, f"{self.indent}{msg}")

    def subprocess(self, msg):
        self._verb_msg(self.lvl_subprocess, f"{self.indent}{self.counter}. {msg}")
        if self.verbose_lvl >= self.lvl_subprocess: self.counter += 1

    def _verb_msg(self, lvl, msg):
        if lvl <= self.verbose_lvl:
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\n[{now_str}] {msg}")
