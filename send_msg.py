class User:
    def __init__(self, host, username, passwd, sender, receivers):
        self.host = host
        self.username = username
        self.__passwd = passwd
        self.sender = sender
        self.receivers = receivers

    def get_passwd(self):
        return self.__passwd


def create_user():
    import pickle
    host = input("please input your email host：\n")
    username = input("please input your username：\n")
    passwd = input("please input your passwd：\n")
    sender = username
    receiver = input("please input your receiver email：\n")
    user = User(host, username, passwd, sender, receiver)
    file = open('output/user.pickle', 'wb')
    pickle.dump(user, file)
    return user


def load_user_account():
    import pickle
    import os
    user_account_path = "./output/user.pickle"
    if not os.path.exists(user_account_path):
        print("No available account exists now. Please create a new one.")
        user = create_user()
    else:
        with open('./output/user.pickle', 'rb') as file:
            user = pickle.load(file)
        print("Already has a account. Your email is: {your_email} and the receiver is: {receiver}".format(
            your_email=user.sender, receiver=user.receivers))
        mode = input(
            "Please enter 1 to load the existing account or\nenter 2 to change e-mail receiver or \nenter 3 to create a new account:\n")
        if mode == '2':
            change_receiver(user)
        elif mode == '3':
            user = create_user()

    return user


def change_receiver(user):
    import pickle
    new_receiver = input("Please enter a new receiver:\n")
    user.receivers = new_receiver
    file = open('output/user.pickle', 'wb')
    pickle.dump(user, file)


def send_email(user, img_path):
    import smtplib
    import time
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.header import Header
    mail_host = user.host
    mail_user = user.username
    mail_pass = user.get_passwd()

    sender = user.sender
    receivers = [user.receivers]
    body = "A suspected fall was detected at {time}. Please check the attached file!"
    now_time = time.strftime("%m/%d/%Y - %H:%M:%S")
    message = MIMEMultipart('related')
    text = MIMEText(body.format(time=now_time), 'plain', 'utf-8')
    message.attach(text)
    message['From'] = sender
    message['To'] = receivers[0]

    file = open(img_path, "rb")
    img_data = file.read()
    file.close()
    img = MIMEImage(img_data)

    img.add_header('Content-Disposition', 'attachment', filename="suspected_img.png")
    message.attach(img)

    subject = 'Warning: Fall Detected!'
    message['Subject'] = Header(subject, 'utf-8')
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("Already sent a msg")
    except smtplib.SMTPException:
        print("An error occurred.")


if __name__ == '__main__':
    user = load_user_account()
    suspected_img_path = "./output/suspected_img.png"
    send_email(user, suspected_img_path)
