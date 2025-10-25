SELECT 'CREATE DATABASE kinitrodb OWNER validator'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'kinitrodb'
)\gexec
