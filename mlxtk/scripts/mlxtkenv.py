from ..settings import load_full_env


def main():
    env = load_full_env()
    for entry in env:
        if isinstance(env[entry], list):
            print("export " + entry + "=\"" +
                  ":".join([str(p) for p in env[entry]]) + ":$" + entry + "\"")
        else:
            print("export " + entry + "=\"" + env[entry] + "\"")


if __name__ == "__main__":
    main()
