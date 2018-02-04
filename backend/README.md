# Backend Description
The backend API is served by Flask through Nginx and uWSGI.

## Backend endpoints

The backend API has the following endpoints and methods:

| Endpoint                  |      Method  |  Options |
|-------------------------- |:------------:|:------|
| /available-endpoints|  GET         |  |
| /api/image     |  POST            | body: image=[{"provider_address"}] |
