# Build date (YYYYMMDD) of the version header and the release header. Stamped
# builds (`--stamp`, set by release-dpc) read the workspace status instead.

DEV_BUILD_DATE = "21990101"

_TOKEN = "@ONEDAL_BUILD_DATE@"

BuildDateInfo = provider(
    doc = "Whether the build is stamped, and the file that holds the date.",
    fields = ["stamped", "file"],
)

def _build_date_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name)
    stamped = ctx.attr.stamp == 1
    if stamped:
        ctx.actions.run_shell(
            inputs = [ctx.info_file],
            outputs = [out],
            command = """
d=$(grep '^STABLE_ONEDAL_BUILD_DATE ' "$1" | cut -d' ' -f2 | tr -d '\r')
case "$d" in
  [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) printf '%s\\n' "$d" > "$2" ;;
  *) echo "$1: no STABLE_ONEDAL_BUILD_DATE, build with --workspace_status_command" >&2; exit 1 ;;
esac
""",
            arguments = [ctx.info_file.path, out.path],
        )
    else:
        ctx.actions.write(out, DEV_BUILD_DATE + "\n")
    return [
        DefaultInfo(files = depset([out])),
        BuildDateInfo(stamped = stamped, file = out),
    ]

build_date = rule(
    implementation = _build_date_impl,
    doc = "The build date: STABLE_ONEDAL_BUILD_DATE when stamped, DEV_BUILD_DATE otherwise.",
    attrs = {
        "stamp": attr.int(
            values = [-1, 0, 1],
            default = -1,
            doc = "1 reads the workspace status. BUILD selects 1 for --stamp; -1 is not stamped.",
        ),
    },
)

def expand_build_date(ctx, template, out, build_date):
    """Writes `out` from `template`, replacing the `@ONEDAL_BUILD_DATE@` token.

    Unstamped builds substitute DEV_BUILD_DATE at analysis, so no action reads the date.
    """
    if not build_date.stamped:
        ctx.actions.expand_template(
            template = template,
            output = out,
            substitutions = {_TOKEN: DEV_BUILD_DATE},
        )
        return
    ctx.actions.run_shell(
        inputs = [template, build_date.file],
        outputs = [out],
        command = 'd=$(cat "$2") && sed "s/@ONEDAL_BUILD_DATE@/$d/g" "$1" > "$3"',
        arguments = [template.path, build_date.file.path, out.path],
    )
