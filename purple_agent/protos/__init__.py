"""
Protobuf package marker -- Purple Agent v7.0

v7.0 [BUG-02 FIX]: This file was identified as a build-blocker in the
v6.0 audit ("protos/__init__.py missing -> import failure") but was never
included in the 14 prerequisite files. Without this marker, the import
`from protos import financial_crime_pb2` can fail in environments where
Python's namespace package resolution does not cover the protos/ directory
(particularly inside Docker containers and some pytest configurations).

This file makes protos/ an explicit regular package.
"""
