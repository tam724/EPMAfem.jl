# run with coverage:

```
> julia --code-coverage=user 
```

then run the specific file, close julia
To analyze with vscode: process + visualize

```
using Coverage
coverage = process_folder()
open("lcov.info", "w") do iousing Coverage
coverage = process_folder()
open("lcov.info", "w") do io
    LCOV.write(io, coverage)
end;
    LCOV.write(io, coverage)
end;
```

