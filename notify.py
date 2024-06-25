import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email(subject, body, recipient="shimomura.teruki174@mail.kyutech.jp", sender_email="shimomura.teruki174@mail.kyutech.jp", sender_password="4bxRLtu2", attachment_path=None):
    # メールの内容を設定
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient
    msg['Subject'] = subject

    # メール本文を追加
    msg.attach(MIMEText(body, 'plain'))

    # 添付ファイルがある場合
    if attachment_path:
        # ファイルを開く
        with open(attachment_path, "rb") as attachment:
            # MIMEBaseを使用してファイルを読み込む
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # ファイルをエンコード
        encoders.encode_base64(part)

        # ヘッダーを追加
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(attachment_path)}",
        )

        # メッセージに添付ファイルを追加
        msg.attach(part)

    # SMTPサーバーに接続
    server = smtplib.SMTP('smtp-mail.outlook.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    
    # メール送信
    server.sendmail(sender_email, recipient, msg.as_string())
    server.quit()