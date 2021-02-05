import argparse
import glob
import smtplib
import ssl
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Tuple, List

import yaml
import os

from utilities import HOME, SUPPORTED_ENVS


def build_attachment(filepath: str, name: str) -> MIMEBase:
    # Open file in binary mode
    with open(filepath, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {name}",
    )
    return part


def get_account_details() -> Tuple[str, str, str]:
    abs_params_dir = os.path.abspath(HOME)
    with open(abs_params_dir + "/gmail.yml", "r") as f:
        account_details = yaml.safe_load(f)
        sender_email = account_details['account']['sender_email']
        receiver_email = account_details['account']['receiver_email']
        password = account_details['account']['password']
    return sender_email, receiver_email, password


def send_email(subject: str, password: str, from_field: str, to_field: str, body: str, attachments: List[MIMEBase] = None) -> None:
    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = from_field
    message["To"] = to_field
    message.attach(MIMEText('{} \n\n'.format(body), "plain"))
    if attachments:
        for attachment in attachments:
            message.attach(attachment)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(from_field, password)
        server.sendmail(from_field, to_field, message.as_string())


def convert_time(time_elapsed_s: float) -> Tuple[float, str]:
    if time_elapsed_s < 60:
        return time_elapsed_s, 's'

    if 60 <= time_elapsed_s < 3600:
        return round(time_elapsed_s / 60, 2), 'min'

    if time_elapsed_s > 3600:
        return round(time_elapsed_s / 3600, 2), 'h'


class MonitorProgress:

    def __init__(self, algo_name: str, env_name: str, results_dir: str,
                 search_type: str, start_search_time: float, param_names_string: str = None,
                 starting_progress_report_number: int = 0):
        self.algo_name = algo_name
        self.env_name = env_name
        self.param_names_string = param_names_string
        self.results_dir = results_dir
        self.progress_report_number = starting_progress_report_number
        self.search_type = search_type
        self.start_search_time = start_search_time

        self.sender_email, self.receiver_email, self.password = get_account_details()

    def send_progress_report(self, time_elapsed: float):
        subject = 'Progress report # {} for experiment {}_{}_{}_{}'.format(
            self.progress_report_number, self.search_type, self.env_name, self.algo_name, self.param_names_string) if self.param_names_string else \
            'Progress report # {} for experiment {}_{}_{}'.format(self.progress_report_number, self.search_type, self.env_name, self.algo_name)

        time_elapsed_unit, unit = convert_time(time_elapsed_s=time_elapsed)
        time_elapsed_unit_global, unit_global = convert_time(time_elapsed_s=(time.time() - self.start_search_time))

        body = 'Time elapsed iteration {} {}. Time elapsed global {} {}\n'.format(
            time_elapsed_unit, unit, time_elapsed_unit_global, unit_global)
        body += 'Documents in results dir {}: \n'.format(self.results_dir)
        for document in os.listdir(self.results_dir):
            body += document + '\n'

        send_email(subject=subject, password=self.password, from_field=self.sender_email, to_field=self.receiver_email, body=body)
        self.progress_report_number += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, default="Test email subject")
    parser.add_argument("--body", type=str, required=True, default="No body")
    parser.add_argument('--experiment_type', choices=['alphatest', 'random'], required=True)
    parser.add_argument('--env_name', choices=SUPPORTED_ENVS, required=True)
    parser.add_argument("--filename_prefix", type=str, required=True, default="filename")

    args, _ = parser.parse_known_args()

    sender_email, receiver_email, password = get_account_details()

    scripts_folder = os.path.join(HOME, 'scripts')
    experiments_folder = os.path.join(scripts_folder, args.experiment_type)
    env_folder = os.path.join(experiments_folder, args.env_name)
    # assuming there is only one file that matches
    output_file = glob.glob(os.path.join(env_folder, "{}*.out".format(args.filename_prefix)))[0]
    error_file = glob.glob(os.path.join(env_folder, "{}*.err".format(args.filename_prefix)))[0]

    attachment_1 = build_attachment(filepath=output_file, name='{}_out.txt'.format(args.filename_prefix))
    attachment_2 = build_attachment(filepath=error_file, name='{}_err.txt'.format(args.filename_prefix))

    attachments = [attachment_1, attachment_2]
    send_email(subject=args.subject, password=password, from_field=sender_email, to_field=receiver_email, body=args.body, attachments=attachments)

    os.remove(output_file)
    os.remove(error_file)
