import requests

def send_line(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'GOqN8UvbQwCcr4F3xx7CtaY5dw3z32M17SK9Y1Jv80x'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)