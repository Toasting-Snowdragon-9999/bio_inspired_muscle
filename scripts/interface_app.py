#!/usr/bin/env python3

"""@brief Terminal-UI entry point for the robot interface application.

Presents a top-level terminal menu (via simple_term_menu) from which the user
can open the simulation command page or exit. ``RobotInterfaceApp`` drives the
menu loop and dispatches to the appropriate command page.
"""

# Default libraries
import argparse
from typing import Optional

# External libraries
from simple_term_menu import TerminalMenu

from command_page.sim_command_page import SimulationCommandPage


class RobotInterfaceApp(object):
    """@brief Top-level terminal application presenting the main interface menu."""
    def __init__(self) -> None:
        """@brief Initialize the main menu options and the simulation command page."""

        # Main menu options
        self.simulation_menu_option = "Simulation options"
        self.current_cursor_index = 0
        self.main_menu_exit = "Exit"

        self.main_menu_options = [self.simulation_menu_option,
                                  self.main_menu_exit
                                  ]
        
        self.simulation_command_page = SimulationCommandPage()


    def show_main_menu(self, cursor_index: int = 0) -> Optional[int]:
        """@brief Display the main menu and return the chosen entry index.

        @param cursor_index: Index the menu cursor starts on.
        @return The selected entry index, or None if the user cancelled.
        """
        menu_title = "Main Menu"
        terminal_menu = TerminalMenu(self.main_menu_options, title=menu_title, cursor_index=cursor_index, show_shortcut_hints=False)
        menu_entry_index = terminal_menu.show()
        return menu_entry_index

    def run_app(self) -> None:
        """@brief Run the main menu loop until the user selects Exit."""

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
        """@brief Clear the current terminal line and print a status message.

        @param message: Status text to display.
        """
        print("\033[2K\r" + message, end="")  # Clear current line and print the message

    def clear_current_line(self) -> None:
        """@brief Clear the current terminal line."""
        print("\033[2K\r", end="")  # Clear the line again


def main() -> None:
    """@brief Construct the RobotInterfaceApp and run its menu loop."""

    app = RobotInterfaceApp()

    app.run_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args()


    main()