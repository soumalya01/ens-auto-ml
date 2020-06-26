#!/bin/bash

mkdir -p ~/.streamlit/
echo "
[general]
email = \"soumalya01@gmail.com\"
" > ~/.streamlit/credentials.toml

echo "
[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml