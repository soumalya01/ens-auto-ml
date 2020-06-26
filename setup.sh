<<<<<<< HEAD
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
=======
mkdir -p ~/.streamlit/

echo "\
[server]\n\n
port = $PORT\n\
enableCORS = false\n\
\n\
>>>>>>> make it better
" > ~/.streamlit/config.toml