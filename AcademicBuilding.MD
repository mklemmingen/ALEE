The next task is to add better logs:

the callerswithtexts need two modular python scripts in their own directory.

1. caller -> methods that can be used by any script in callerswithtexts root to call the orchestrator and get the texts.

2. logger -> methods that can be used by any script in callerswithtexts root to log messages in _logs with iso8601 timestamps.
as well as -> methods that can be called by analyzing tools to access the logger caller logs to build meaningful data points.

After these are done, they need to be added to each python script in callerswithtexts root, so they all share the logger and caller methods.

