import json
import random
from time import sleep

import requests

from .auth import Authentication


class UMLSClient:
    def __init__(self, apikey, version):
        self.auth_client = Authentication(apikey=apikey)
        self.version = version

    def get_ticket(self):
        tgt = self.auth_client.gettgt()
        ticket = self.auth_client.getst(tgt)
        return ticket

    def traverse_rest_resource(self, endpoint, repeat_count=0):
        if not endpoint or endpoint.lower() == 'none':
            return None

        params = {'pageSize': 10, 'language': 'ENG'}

        # generate a new service ticket
        ticket = self.get_ticket()
        # query = { 'ticket':ticket }
        params['ticket'] = ticket
        r = requests.get(endpoint, params=params)
        r.encoding = 'utf-8'
        if r.status_code != 200:
            # print("Failed: ", r.status_code, "Sleeping for 5 secs")
            # print(r.text)
            sleep(random.random() * 3 + 2)
            if repeat_count > 2:
                print("Slept after failure: ", r.status_code)
            if repeat_count < 4:
                return self.traverse_rest_resource(endpoint, repeat_count=repeat_count + 1)
            else:
                raise RuntimeError(endpoint)

        # print(r.text)
        items = json.loads(r.text)
        return items["result"]

    def search(self, string, repeat_count, **extra_args):
        """Run a search query on the UMLS REST API

        Parameters:
            string    -- the string to search for (e.g., "diabetic foot")
            apikey    -- account-linked API key for authentication
            version   -- UMLS release to search (default '2016AB')
            max_pages -- maximum number of result pages to fetch (default 5; None for unlimited)
            **        -- additional keyword args are passed into the API query
        """
        uri = "https://uts-ws.nlm.nih.gov"
        version = self.version
        content_endpoint = "/rest/search/" + version

        # get authentication granting ticket for the session
        #     AuthClient = Authentication(apikey)
        #     tgt = AuthClient.gettgt()
        page_number = 0

        results = []
        # generate a new service ticket for each page if needed
        ticket = self.get_ticket()

        page_number += 1
        query = {'string': string, 'ticket': ticket, 'pageNumber': page_number}
        for (k, v) in extra_args.items():
            query[k] = v

        r = requests.get(uri + content_endpoint, params=query)
        r.encoding = 'utf-8'
        if r.status_code != 200:
            sleep(random.random() * 3 + 2)
            if repeat_count > 2:
                print("Slept after failure: ", r.status_code)
            if repeat_count < 4:
                return self.search(string, repeat_count + 1, **extra_args)
            else:
                raise RuntimeError(string + r.text)
        items = json.loads(r.text)
        json_data = items["result"]

        # break option 2: no more results
        if json_data["results"][0]["ui"] == "NONE":
            pass
        else:
            results.extend(json_data["results"])
        return results
