from agent import train


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_name = input('Enter the name of the model to load: ')
    if file_name == "":
        train()
    else:
        train(model_path=file_name)
    # train()
