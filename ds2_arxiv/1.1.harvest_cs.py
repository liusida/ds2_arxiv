import time, os
import xmltodict # to convert the raw metadata from xml format to dict
# import pandas as pd # final format of stored data
from sickle import Sickle # to retrieve data from the OAI arxiv interface
from requests import HTTPError
from datetime import datetime, timedelta

def main(from_date, to_date):
    connection = Sickle('http://export.arxiv.org/oai2')

    while True:
        try:
            print('Getting papers...')
            data = connection.ListRecords(**{'metadataPrefix': 'arXiv', 'from': from_date, 'until': to_date, 'ignore_deleted': True, 'set': 'cs'})
            print('Papers retrieved.')
            break
        except HTTPError:
            print(f"wait for 3 seconds.")
            time.sleep(3)
        except Exception as e:
            print(f'Other exception {e}')
            break
            
    iters = 0

    while True:
        try:
            record_xml = data.next().raw
            record_dict = xmltodict.parse(record_xml, process_namespaces=False)['record']['metadata']['arXiv']
            arxiv_id = record_dict['id']
            arxiv_id = arxiv_id.split("/")[-1]
            if arxiv_id.find(".")==-1:
                print(f"Format not right {arxiv_id}")
                continue
            path = f"data/harvest/{arxiv_id}.xml"
            if os.path.exists(path):
                # print(f"exist {path}")
                continue
            print(f"writing {path}")
            with open(path, "w") as f:
                f.write(data.next().raw)
            errors = 0
            iters +=1
            
            if iters % 900 == 0:
                time.sleep(3)
                print('On iter', iters)
        
        except AttributeError:
            print(f'ERROR! {errors}\n')
            errors +=1
        except HTTPError:
            print(f"wait for 3 seconds.")
            time.sleep(3)
        except StopIteration:
            print('On iter', iters)
            print('\nDONE!')
            break
        except Exception as e:
            print(f'Other exception {e}')
            break

if __name__=="__main__":
    start_date = datetime.today()
    d_date = timedelta(days=30)
    start_date = start_date -d_date
    while True:
        start_date_string = start_date.strftime('%Y-%m-%d')
        end_date = start_date - d_date
        end_date_string = end_date.strftime('%Y-%m-%d')
        if end_date_string[:4]=='1999':
            break
        print(end_date_string, start_date_string)

        main(end_date_string, start_date_string)
        
        start_date = end_date
        time.sleep(3)