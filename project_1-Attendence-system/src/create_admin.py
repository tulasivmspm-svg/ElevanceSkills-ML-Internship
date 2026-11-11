
from getpass import getpass
from database import add_user

username = input("Enter admin username: ").strip()
password = getpass("Create admin password: ")
confirm_password = getpass("Confirm password: ")

if not username or not password:
    print("Username and password cannot be empty.")

elif password != confirm_password:
    print("Passwords do not match.")

else:
    success = add_user(username, password, "admin")

    if success:
        print("Admin account created successfully!")
    else:
        print("This username already exists.")
