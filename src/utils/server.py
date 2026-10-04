import socket

START_COMMAND = "node pokemon-showdown start --no-security"


def check_showdown_server(host: str = "localhost", port: int = 8000):
    """
    Exits with instructions if there's no Showdown server, instead of waiting for
    poke-env's challenge timeout.
    """
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex((host, port)) != 0:
            raise SystemExit(
                f"No Showdown server on {host}:{port}. Start one from your "
                f"pokemon-showdown folder with: {START_COMMAND}"
            )
