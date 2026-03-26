#!/usr/bin/env python3

# Default libraries
import argparse
import time
from typing import Optional

# External libraries
from simple_term_menu import TerminalMenu

from command_page.sim_command_page import SimulationCommandPage


class RobotInterfaceApp(object):
    def __init__(self) -> None:

        # Main menu options
        self.simulation_menu_option = "Simulation options"
        self.current_cursor_index = 0
        self.main_menu_exit = "Exit"

        self.main_menu_options = [self.simulation_menu_option,
                                  self.main_menu_exit
                                  ]
        
        self.simulation_command_page = SimulationCommandPage()


    def show_main_menu(self, cursor_index: int = 0) -> Optional[int]:
        menu_title = "Main Menu"
        terminal_menu = TerminalMenu(self.main_menu_options, title=menu_title, cursor_index=cursor_index, show_shortcut_hints=False)
        menu_entry_index = terminal_menu.show()
        return menu_entry_index

    def run_app(self) -> None:

        while True:
            idx = self.show_main_menu(self.current_cursor_index)
            if idx is None:
                main_option = self.main_menu_exit
            else:
                self.current_cursor_index = idx
                main_option = self.main_menu_options[self.current_cursor_index]

            if main_option == self.simulation_menu_option:
                self.simulation_command_page.run_simulation_command_menu()
            elif main_option == self.main_menu_exit:
                self.display_status("Exiting...")
                break

            print("\033[2K\r", end="")  # Clear the line again

    # Helper function to print status and clear after menu
    def display_status(self, message: str) -> None:
        print("\033[2K\r" + message, end="")  # Clear current line and print the message

    def clear_current_line(self) -> None:
        print("\033[2K\r", end="")  # Clear the line again


def main() -> None:

    app = RobotInterfaceApp()

    app.run_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args()


    main()