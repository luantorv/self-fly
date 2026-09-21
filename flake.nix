{
  description = "SELF-FLY: experimento computacional de estabilidad de política y evento autorreferencial";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3.withPackages (ps: with ps; [
          tkinter
          numpy
          matplotlib
          pytest
        ]);

        headlessScript = pkgs.writeShellScriptBin "self-fly-headless" ''
          exec ${python}/bin/python ${self}/run_experiment.py "$@"
        '';

        guiScript = pkgs.writeShellScriptBin "self-fly-gui" ''
          exec ${python}/bin/python ${self}/run_gui.py "$@"
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
          ];

          shellHook = ''alias py="python"'';
        };

        apps = {
          default = {
            type = "app";
            program = "${headlessScript}/bin/self-fly-headless";
          };
          headless = {
            type = "app";
            program = "${headlessScript}/bin/self-fly-headless";
          };
          gui = {
            type = "app";
            program = "${guiScript}/bin/self-fly-gui";
          };
        };
      }
    );
}
