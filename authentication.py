import easygui
import os

# def findPIN():
#     dir = "/home/pi/Desktop/human-detection-main"
#     fname = "passcode.txt"

#     if os.path.isfile(os.path.join(dir, fname)):
#         print(f"{fname} exists in {dir}")
#     else:
#         print(f"{fname} does not exist in {dir}")
#         random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
#         formatted_string = '-'.join([random_string[i:i+4].upper() for i in range(0, len(random_string), 4)])
        
#         fpath = os.path.join(dir, fname)

#         with open(fpath, "w") as file:
#             file.write("Product Code: " + formatted_string)
#         print(f"{fname} has been created and saved in {dir}.")

# Function to validate the PIN
def validate_pin(pin):
    dir = "/home/pi/Desktop/human-detection-main"
    fname = "passcode.txt"

    if pin == None:
        return False
    elif len(pin) == 4 and pin.isdigit():
        fpath = os.path.join(dir, fname)

        with open(fpath, "w") as file:
            file.write(pin)
            
        return True
    # if len(pin) == 4 and pin.isdigit():
    #     fpath = os.path.join(dir, fname)

    #     with open(fpath, "w") as file:
    #         file.write(pin)
            
    #     return True
    # return False

# Prompt the user to enter a PIN
pin = easygui.enterbox("Enter a 4-digit PIN:", title="PIN Entry")

# Validate the entered PIN
while not validate_pin(pin):
    easygui.msgbox("Invalid PIN. Please enter a 4-digit number.", title="Invalid PIN")
    pin = easygui.enterbox("Enter a 4-digit PIN:", title="PIN Entry")

# Display the entered PIN
easygui.msgbox(f"PIN entered: {pin}", title="PIN Confirmation")
