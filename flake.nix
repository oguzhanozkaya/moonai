{
  description = "cpp flake template";
  inputs = { nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; };
  outputs = { self, nixpkgs }:
  let
    supportedSystems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
  in {
    devShells = forAllSystems (system: 
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            gnumake
            gcc
            clang
            vcpkg
            clang-tools
            cppcheck
            llvm
          ];

          CC = "${pkgs.gcc}/bin/gcc";
          CXX = "${pkgs.gcc}/bin/g++";

          shellHook = ''
            echo "Project Packages and environment loaded for ${system}."
          '';
        };
      }
    );
  };
}
