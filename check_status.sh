#!/bin/bash
url=`cat server-addr`
curl -H 'Content-Type: application/json' $url"get_status"
