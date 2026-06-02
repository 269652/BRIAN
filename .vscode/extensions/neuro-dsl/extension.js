const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const LANGUAGE_ID = 'neuro';
const OUTPUT_CHANNEL = vscode.window.createOutputChannel('NeuroSLM DSL');

// Collect diagnostics from linter output
const diagnosticCollection = vscode.languages.createDiagnosticCollection('neuro');

// Linting keywords for autocomplete
const KEYWORDS = [
  'architecture', 'neurotransmitter', 'population', 'export', 'synapse',
  'import', 'from', 'training', 'dynamics', 'function', 'mechanisms',
  'count', 'equation', 'ode', 'weight', 'dynamics', 'timescale', 'capacity'
];

const BUILTIN_FUNCTIONS = [
  'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'sigmoid',
  'ReLU', 'silu', 'gelu', 'swiglu', 'matmul', 'linear', 'rmsnorm',
  'causal_self_attention', 'embedding', 'softmax', 'dropout', 'layer_norm'
];

const BUILTIN_VARS = [
  'x', 'y', 's', 'V', 'd_sem', 'dt', 'tau', 'weight',
  'x_pre', 'x_post', 'e', 'pi', 'nan', 'inf'
];

/**
 * Search for a reference (@name) in an imported file
 */
function searchImportedFile(currentFileUri, importPath, reference) {
  const baseDir = path.dirname(currentFileUri.fsPath);
  const refName = reference.slice(1); // remove @ prefix

  // Try candidate paths
  const candidates = [
    path.join(baseDir, `${importPath}.neuro`),
    path.join(baseDir, importPath, 'index.neuro'),
    path.join(baseDir, importPath)
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      try {
        const content = fs.readFileSync(candidate, 'utf8');
        const patterns = [
          new RegExp(`\\b(export\\s+)?(equation|dynamics|function|population)\\s+${refName}\\s*[{:]`),
          new RegExp(`\\b(equation|dynamics|function)\\s+${refName}\\s*\\{`)
        ];

        for (const pattern of patterns) {
          const match = pattern.exec(content);
          if (match) {
            const doc = vscode.workspace.textDocuments.find(d => d.uri.fsPath === candidate);
            if (doc) {
              const matchPos = match.index;
              const lineNum = doc.positionAt(matchPos).line;
              return new vscode.Location(
                vscode.Uri.file(candidate),
                new vscode.Position(lineNum, 0)
              );
            }
          }
        }
      } catch (e) {
        // Silently skip unreadable files
      }
    }
  }

  return null;
}

exports.activate = function(context) {
  OUTPUT_CHANNEL.appendLine('NeuroSLM DSL extension activating...');

  // Register definition provider for @references (Ctrl+Click, F12, Go to Definition)
  context.subscriptions.push(
    vscode.languages.registerDefinitionProvider(LANGUAGE_ID, {
      provideDefinition(document, position) {
        const word = document.getWordRangeAtPosition(position, /@[\w]+/);
        if (!word) return null;

        const reference = document.getText(word); // includes @ prefix
        const text = document.getText();

        // Search for definition: "export equation|dynamics|function|population NAME {" or similar
        const patterns = [
          new RegExp(`\\b(export\\s+)?(equation|dynamics|function|population)\\s+${reference.slice(1)}\\s*[{:]`, 'g'),
          new RegExp(`\\b(equation|dynamics|function)\\s+${reference.slice(1)}\\s*\\{`, 'g')
        ];

        for (const pattern of patterns) {
          const match = pattern.exec(text);
          if (match) {
            const matchPos = match.index;
            const lineNum = document.positionAt(matchPos).line;
            const charNum = document.positionAt(matchPos).character;
            return new vscode.Location(
              document.uri,
              new vscode.Position(lineNum, charNum)
            );
          }
        }

        // If not found in current file, search in imported files
        const importMatches = [...text.matchAll(/import\s*\{[^}]*\}\s*from\s*["'](@?[^"']+)["']/g)];
        for (const importMatch of importMatches) {
          const importPath = importMatch[1].replace(/^@\//, '');
          const importedDef = searchImportedFile(document.uri, importPath, reference);
          if (importedDef) return importedDef;
        }

        return null;
      }
    })
  );

  // Register hover provider for @references
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(LANGUAGE_ID, {
      provideHover(document, position) {
        const word = document.getWordRangeAtPosition(position, /@[\w]+/);
        if (!word) return null;

        const reference = document.getText(word); // includes @ prefix
        const refName = reference.slice(1);
        const text = document.getText();

        // Search for the definition in current file
        const patterns = [
          { pattern: new RegExp(`export\\s+equation\\s+${refName}\\s*\\{([^}]*)\\}`, 's'), type: 'equation' },
          { pattern: new RegExp(`equation\\s+${refName}\\s*\\{([^}]*)\\}`, 's'), type: 'equation' },
          { pattern: new RegExp(`export\\s+dynamics\\s+${refName}\\s*\\{([^}]*)\\}`, 's'), type: 'dynamics' },
          { pattern: new RegExp(`dynamics\\s+${refName}\\s*\\{([^}]*)\\}`, 's'), type: 'dynamics' },
        ];

        for (const { pattern, type } of patterns) {
          const match = pattern.exec(text);
          if (match) {
            const content = match[1];
            // Extract formula if it's an equation
            let formula = '';
            if (type === 'equation') {
              const formulaMatch = content.match(/formula:\s*"([^"]*)"/);
              if (formulaMatch) {
                formula = formulaMatch[1];
              }
            }

            // Extract params
            const paramsMatch = content.match(/params:\s*\[([^\]]*)\]/);
            const params = paramsMatch ? paramsMatch[1].trim() : '';

            let hoverText = `**${type}** \`${refName}\`\n\n`;
            if (params) {
              hoverText += `**params:** ${params}\n\n`;
            }
            if (formula) {
              hoverText += `**formula:** \`${formula}\``;
            }

            return new vscode.Hover(new vscode.MarkdownString(hoverText));
          }
        }

        // If not found in current file, try imported files
        const importMatches = [...text.matchAll(/import\s*\{[^}]*\}\s*from\s*["'](@?[^"']+)["']/g)];
        for (const importMatch of importMatches) {
          const importPath = importMatch[1].replace(/^@\//, '');
          const baseDir = path.dirname(document.uri.fsPath);
          const candidates = [
            path.join(baseDir, `${importPath}.neuro`),
            path.join(baseDir, importPath, 'index.neuro'),
            path.join(baseDir, importPath)
          ];

          for (const candidate of candidates) {
            if (fs.existsSync(candidate)) {
              try {
                const importedContent = fs.readFileSync(candidate, 'utf8');
                for (const { pattern, type } of patterns) {
                  const match = pattern.exec(importedContent);
                  if (match) {
                    const content = match[1];
                    let formula = '';
                    if (type === 'equation') {
                      const formulaMatch = content.match(/formula:\s*"([^"]*)"/);
                      if (formulaMatch) {
                        formula = formulaMatch[1];
                      }
                    }

                    const paramsMatch = content.match(/params:\s*\[([^\]]*)\]/);
                    const params = paramsMatch ? paramsMatch[1].trim() : '';

                    let hoverText = `**${type}** \`${refName}\` _(imported)_\n\n`;
                    if (params) {
                      hoverText += `**params:** ${params}\n\n`;
                    }
                    if (formula) {
                      hoverText += `**formula:** \`${formula}\``;
                    }

                    return new vscode.Hover(new vscode.MarkdownString(hoverText));
                  }
                }
              } catch (e) {
                // Silently skip unreadable files
              }
            }
          }
        }

        return null;
      }
    })
  );

  // Register code action provider for autofix
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(LANGUAGE_ID, {
      provideCodeActions(document, range, context) {
        const actions = [];

        // Check diagnostics in this range
        for (const diagnostic of context.diagnostics) {
          if (diagnostic.code === 'repeated-equation') {
            const action = new vscode.CodeAction(
              'Extract as equation definition',
              vscode.CodeActionKind.QuickFix
            );
            action.command = {
              title: 'Extract as equation definition',
              command: 'neuro-dsl.extractEquation',
              arguments: [document, diagnostic]
            };
            action.diagnostics = [diagnostic];
            actions.push(action);
          } else if (diagnostic.code === 'arch-move-equations' || diagnostic.code === 'equation-in-arch') {
            const action = new vscode.CodeAction(
              'Move equation to lib/equations.neuro',
              vscode.CodeActionKind.QuickFix
            );
            action.command = {
              title: 'Move equation to lib/equations.neuro',
              command: 'neuro-dsl.extractEquationToLib',
              arguments: [document, diagnostic]
            };
            action.diagnostics = [diagnostic];
            actions.push(action);
          }
        }

        return actions;
      }
    })
  );

  // Register command for moving equations to lib
  context.subscriptions.push(
    vscode.commands.registerCommand('neuro-dsl.moveEquationsToLib', async (document) => {
      const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
      if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
      }

      const libDir = path.join(workspaceFolder.uri.fsPath, 'lib');
      const equationsFile = path.join(libDir, 'equations.neuro');

      // Ensure lib directory exists
      if (!fs.existsSync(libDir)) {
        fs.mkdirSync(libDir, { recursive: true });
      }

      // Extract all equation definitions from current file
      const text = document.getText();
      const equationMatches = [...text.matchAll(/^(export\s+)?equation\s+\w+\s*\{[^}]*\}/gm)];

      if (equationMatches.length === 0) {
        vscode.window.showInformationMessage('No equation definitions found in this file');
        return;
      }

      // Get existing equations from lib file
      let existingEquations = '';
      if (fs.existsSync(equationsFile)) {
        existingEquations = fs.readFileSync(equationsFile, 'utf8');
      }

      // Collect all equations to write
      const equationTexts = equationMatches.map(m => m[0]);
      const combinedEquations = existingEquations + (existingEquations.trim() ? '\n\n' : '') + equationTexts.join('\n\n') + '\n';

      // Write to lib/equations.neuro
      fs.writeFileSync(equationsFile, combinedEquations, 'utf8');

      // Remove equations from current file and add import statement
      let newText = text;

      // Remove equation definitions
      equationTexts.forEach(eq => {
        newText = newText.replace(eq + '\n\n', '').replace(eq + '\n', '').replace(eq, '');
      });

      // Add import statement at the top (after architecture block if present)
      const archMatch = newText.match(/architecture\s+\w+\s*\{[^}]*\}/s);
      let insertPos = 0;
      if (archMatch) {
        insertPos = archMatch.index + archMatch[0].length + 1;
      }

      const importStatement = `import { ${equationMatches.map(m => {
        const nameMatch = m[0].match(/equation\s+(\w+)/);
        return nameMatch ? nameMatch[1] : '';
      }).filter(Boolean).join(', ')} } from "@/lib/equations"\n\n`;

      newText = newText.slice(0, insertPos) + importStatement + newText.slice(insertPos);

      // Apply changes
      const edit = new vscode.WorkspaceEdit();
      const endPos = document.positionAt(text.length);
      edit.replace(document.uri, new vscode.Range(new vscode.Position(0, 0), endPos), newText);
      await vscode.workspace.applyEdit(edit);

      // Open the lib/equations.neuro file
      const libUri = vscode.Uri.file(equationsFile);
      await vscode.window.showTextDocument(libUri);

      vscode.window.showInformationMessage(`✓ Moved ${equationTexts.length} equation(s) to lib/equations.neuro`);
    })
  );

  // Register command for extracting a single equation to lib (from diagnostic)
  context.subscriptions.push(
    vscode.commands.registerCommand('neuro-dsl.extractEquationToLib', async (document, diagnostic) => {
      const text = document.getText();
      const line = document.lineAt(diagnostic.range.start.line);

      // Extract equation definition from current line
      const equationMatch = line.text.match(/\b(export\s+)?equation\s+(\w+)\s*\{/);
      if (!equationMatch) {
        vscode.window.showErrorMessage('Could not parse equation definition');
        return;
      }

      const equationName = equationMatch[2];

      // Find the full equation block (handle multi-line)
      const startLine = diagnostic.range.start.line;
      let endLine = startLine;
      let braceCount = (line.text.match(/{/g) || []).length - (line.text.match(/}/g) || []).length;

      while (braceCount > 0 && endLine < document.lineCount - 1) {
        endLine++;
        const nextLine = document.lineAt(endLine).text;
        braceCount += (nextLine.match(/{/g) || []).length;
        braceCount -= (nextLine.match(/}/g) || []).length;
      }

      // Extract full equation block
      const equationStart = document.lineAt(startLine).range.start;
      const equationEnd = document.lineAt(endLine).range.end;
      const fullEquationText = document.getText(new vscode.Range(equationStart, equationEnd));

      // Get or create lib/equations.neuro
      const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
      if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
      }

      const libDir = path.join(workspaceFolder.uri.fsPath, 'architectures', path.basename(path.dirname(document.uri.fsPath)), 'lib');
      const equationsFile = path.join(libDir, 'equations.neuro');

      // Ensure lib directory exists
      if (!fs.existsSync(libDir)) {
        fs.mkdirSync(libDir, { recursive: true });
      }

      // Append equation to lib file
      let existingContent = '';
      if (fs.existsSync(equationsFile)) {
        existingContent = fs.readFileSync(equationsFile, 'utf8');
      }

      fs.writeFileSync(
        equationsFile,
        existingContent + (existingContent.trim() ? '\n\n' : '') + fullEquationText + '\n',
        'utf8'
      );

      // Remove equation from current file and add import
      const edit = new vscode.WorkspaceEdit();

      // Delete equation from arch.neuro
      edit.delete(document.uri, new vscode.Range(
        new vscode.Position(startLine, 0),
        new vscode.Position(endLine + 1, 0)
      ));

      // Add import at top (after architecture block)
      const archMatch = text.match(/architecture\s+\w+\s*\{[^}]*\}/s);
      let insertLine = 0;
      if (archMatch) {
        insertLine = text.substring(0, archMatch.index + archMatch[0].length).split('\n').length;
      }

      const importStatement = `import { ${equationName} } from "@/lib/equations"\n`;
      edit.insert(document.uri, new vscode.Position(insertLine, 0), importStatement + '\n');

      await vscode.workspace.applyEdit(edit);

      // Open lib file
      const libUri = vscode.Uri.file(equationsFile);
      await vscode.window.showTextDocument(libUri);

      vscode.window.showInformationMessage(`✓ Moved equation '${equationName}' to lib/equations.neuro`);
    })
  );

  // Register command for extracting equations
  context.subscriptions.push(
    vscode.commands.registerCommand('neuro-dsl.extractEquation', async (document, diagnostic) => {
      // Get equation name from user
      const equationName = await vscode.window.showInputBox({
        prompt: 'Enter equation name (e.g., standard_synapse)',
        placeHolder: 'equation_name',
        validateInput: (value) => {
          if (!value.match(/^[a-z_][a-z0-9_]*$/i)) {
            return 'Name must be a valid identifier (letters, numbers, underscores)';
          }
          return null;
        }
      });

      if (!equationName) return;

      const text = document.getText();
      const lines = text.split('\n');

      // Extract the equation from the diagnostic message
      const match = text.match(/formula: "([^"]*)"/);
      if (!match) {
        vscode.window.showErrorMessage('Could not extract equation formula');
        return;
      }

      const formula = match[1];

      // Find all parameters in the formula
      const params = [];
      const paramMatch = formula.match(/\b([a-zA-Z_]\w*)\b/g);
      if (paramMatch) {
        const builtins = new Set(['ReLU', 'sigmoid', 'tanh', 'sin', 'cos', 'exp', 'log', 'sqrt', 'matmul', 'x', 'y', 's', 'V', 'dt', 'pi', 'e']);
        paramMatch.forEach(p => {
          if (!builtins.has(p) && !params.includes(p)) {
            params.push(p);
          }
        });
      }

      // Create equation definition
      const equationDef = `export equation ${equationName} {\n    params: [${params.join(', ')}],\n    formula: "${formula}"\n}\n`;

      // Insert equation definition at top of file
      const insertPos = new vscode.Position(0, 0);
      const edit = new vscode.WorkspaceEdit();
      edit.insert(document.uri, insertPos, equationDef + '\n');

      // Replace all repeated equations with reference
      const fullText = document.getText();
      const replacementEdit = fullText.replace(new RegExp(`equation: "${formula.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g'), `equation: @${equationName}`);

      edit.replace(
        document.uri,
        new vscode.Range(new vscode.Position(0, 0), new vscode.Position(lines.length, 0)),
        equationDef + '\n' + replacementEdit
      );

      await vscode.workspace.applyEdit(edit);
      vscode.window.showInformationMessage(`✓ Extracted equation '${equationName}' and replaced ${(replacementEdit.match(new RegExp(`@${equationName}`, 'g')) || []).length} occurrences`);
    })
  );

  // Register autocomplete provider
  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(LANGUAGE_ID, {
      provideCompletionItems(document, position) {
        const linePrefix = document.lineAt(position).text.substr(0, position.character);
        const wordMatch = linePrefix.match(/\b([a-zA-Z_][a-zA-Z0-9_]*)$/);
        const word = wordMatch ? wordMatch[1] : '';

        const completions = [];

        // Add keywords
        KEYWORDS.forEach(keyword => {
          if (keyword.startsWith(word)) {
            const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
            completions.push(item);
          }
        });

        // Add built-in functions with snippet
        BUILTIN_FUNCTIONS.forEach(fn => {
          if (fn.startsWith(word)) {
            const item = new vscode.CompletionItem(fn, vscode.CompletionItemKind.Function);
            item.insertText = new vscode.SnippetString(fn + '($0)');
            completions.push(item);
          }
        });

        // Add built-in variables
        BUILTIN_VARS.forEach(v => {
          if (v.startsWith(word)) {
            const item = new vscode.CompletionItem(v, vscode.CompletionItemKind.Variable);
            completions.push(item);
          }
        });

        // Extract populations from current file
        const text = document.getText();
        const popMatches = text.matchAll(/(?:export\s+)?population\s+(\w+)/g);
        for (const match of popMatches) {
          const popName = match[1];
          if (popName.startsWith(word)) {
            const item = new vscode.CompletionItem(popName, vscode.CompletionItemKind.Class);
            completions.push(item);
          }
        }

        return completions;
      }
    })
  );

  // Lint on open
  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        OUTPUT_CHANNEL.appendLine(`Linting on open: ${doc.fileName}`);
        lintFile(doc);
      }
    })
  );

  // Lint on save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (doc.languageId === LANGUAGE_ID) {
        OUTPUT_CHANNEL.appendLine(`Linting on save: ${doc.fileName}`);
        lintFile(doc);
      }
    })
  );

  // Lint on change (debounced)
  let changeTimeout;
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (event.document.languageId === LANGUAGE_ID) {
        clearTimeout(changeTimeout);
        changeTimeout = setTimeout(() => {
          lintFile(event.document);
        }, 1500);
      }
    })
  );

  OUTPUT_CHANNEL.appendLine('✓ NeuroSLM DSL extension activated successfully');
};

/**
 * Run the Python linter on a document
 */
function lintFile(document) {
  const filePath = document.uri.fsPath;
  const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);

  if (!workspaceFolder) {
    OUTPUT_CHANNEL.appendLine('No workspace folder found');
    return;
  }

  const pythonCmd = findPythonExecutable(workspaceFolder);
  OUTPUT_CHANNEL.appendLine(`Using Python: ${pythonCmd}`);

  if (!pythonCmd) {
    OUTPUT_CHANNEL.appendLine('Error: Python executable not found');
    return;
  }

  // Run linter
  const proc = spawn(pythonCmd, [
    '-m', 'neuroslm.dsl.neuro_linter',
    filePath,
    '--json'
  ], {
    cwd: workspaceFolder.uri.fsPath
  });

  let output = '';
  let errorOutput = '';

  proc.stdout.on('data', (data) => {
    output += data.toString();
  });

  proc.stderr.on('data', (data) => {
    errorOutput += data.toString();
  });

  proc.on('error', (err) => {
    OUTPUT_CHANNEL.appendLine(`✗ Linter process error: ${err.message}`);
    OUTPUT_CHANNEL.appendLine('This usually means Python executable not found');
    diagnosticCollection.set(document.uri, []);
  });

  proc.on('close', (code) => {
    const diags = [];

    if (code === 0 || code === 1) {
      // code 0 = success, code 1 = had errors/warnings (both are ok)
      try {
        if (output.trim()) {
          const results = JSON.parse(output);
          if (Array.isArray(results)) {
            OUTPUT_CHANNEL.appendLine(`✓ Linter found ${results.length} diagnostic(s)`);
            results.forEach(result => {
              const range = new vscode.Range(
                new vscode.Position(result.line - 1, Math.max(0, result.col - 1)),
                new vscode.Position(result.line - 1, Math.min(result.col + 10, 999))
              );

              const severity = {
                'error': vscode.DiagnosticSeverity.Error,
                'warning': vscode.DiagnosticSeverity.Warning,
                'info': vscode.DiagnosticSeverity.Information
              }[result.severity] || vscode.DiagnosticSeverity.Information;

              const diag = new vscode.Diagnostic(
                range,
                `[${result.code}] ${result.message}`,
                severity
              );
              diag.source = 'neuro-linter';
              diags.push(diag);
            });
          }
        } else {
          OUTPUT_CHANNEL.appendLine('✓ Linter passed (no issues)');
        }
      } catch (e) {
        OUTPUT_CHANNEL.appendLine(`✗ JSON parse error: ${e.message}`);
        OUTPUT_CHANNEL.appendLine(`Output was: ${output.substring(0, 500)}`);
      }
    } else if (code === null) {
      OUTPUT_CHANNEL.appendLine(`✗ Linter killed or crashed`);
    } else {
      OUTPUT_CHANNEL.appendLine(`✗ Linter failed with code ${code}`);
      OUTPUT_CHANNEL.appendLine(`stderr: ${errorOutput.substring(0, 500)}`);
    }

    diagnosticCollection.set(document.uri, diags);
  });
}

/**
 * Find Python executable in the workspace venv
 */
function findPythonExecutable(workspaceFolder) {
  const basePath = workspaceFolder.uri.fsPath;
  OUTPUT_CHANNEL.appendLine(`Workspace path: ${basePath}`);

  // Try venv locations in order
  const candidates = [
    path.join(basePath, '.venv', 'Scripts', 'python.exe'),
    path.join(basePath, '.venv', 'bin', 'python'),
    path.join(basePath, 'venv', 'Scripts', 'python.exe'),
    path.join(basePath, 'venv', 'bin', 'python'),
  ];

  for (const candidate of candidates) {
    OUTPUT_CHANNEL.appendLine(`Checking: ${candidate}`);
    if (fs.existsSync(candidate)) {
      OUTPUT_CHANNEL.appendLine(`✓ Found Python at: ${candidate}`);
      return candidate;
    }
  }

  OUTPUT_CHANNEL.appendLine('⚠ Venv Python not found, falling back to system Python');
  OUTPUT_CHANNEL.appendLine('This will fail if system Python < 3.7');
  return process.platform === 'win32' ? 'python.exe' : 'python3';
}

exports.deactivate = function() {
  diagnosticCollection.clear();
  diagnosticCollection.dispose();
  OUTPUT_CHANNEL.dispose();
};
