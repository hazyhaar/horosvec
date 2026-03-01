.PHONY: build test bench

build:
	CGO_ENABLED=0 go build ./...

test:
	go test -race -v -count=1 -timeout 120s ./...

bench:
	go test -bench=. -benchmem -count=1 -timeout 300s ./...
